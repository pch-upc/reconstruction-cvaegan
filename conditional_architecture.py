import torch
# from torchsummary import summary
  
class Encoder(torch.nn.Module):
    def __init__(self, z_dim, batch_size, latent, device):
        
        super(Encoder, self).__init__()
        # dim_in = 128
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.latent = latent
        self.device = device

        self.encoder_conv = torch.nn.Sequential(
            
            # dim: Bx64x64x64
            torch.nn.Conv3d(in_channels=2, out_channels=4, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            
            # dim: Bx32x32x32
            torch.nn.Conv3d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
      
            # dim: Bx16x16x16
            torch.nn.Conv3d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2)
            
        )

        self.encoder_fc1 = torch.nn.Sequential(
            torch.nn.Linear(16*z_dim*z_dim*z_dim, latent), 
            torch.nn.LeakyReLU(0.2)
        )
        
        self.encoder_fc2 = torch.nn.Sequential(
            torch.nn.Linear(16*z_dim*z_dim*z_dim, latent), 
            torch.nn.LeakyReLU(0.2)
        )         
    
    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(self.device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self,x,c):
        c = c.reshape(self.batch_size,1,1,1,1).repeat(1,1,self.z_dim*8,self.z_dim*8,self.z_dim*8)
        con = torch.cat([x, c], 1)
        out1, out2 = self.encoder_conv(con), self.encoder_conv(con)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logvar = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logvar)
        return z,mean,logvar
    
    
class Generator(torch.nn.Module):
    def __init__(self, z_dim, batch_size, latent, device):

        super(Generator, self).__init__()
        
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.latent = latent
        self.device = device

        self.decoder_fc = torch.nn.Sequential(
            torch.nn.Linear(latent+latent, 16*z_dim*z_dim*z_dim), 
            torch.nn.LeakyReLU(0.2)
        )

        self.decoder_deconv = torch.nn.Sequential(

            # dim: Bx32x32x32
            torch.nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),

            # dim: Bx64x64x64
            torch.nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            
            # dim: Bx128x128x128
            torch.nn.ConvTranspose3d(in_channels=4, out_channels=1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self,z,c):
        c = c.reshape(self.batch_size,1).repeat(1,self.latent)
        # c = c.reshape(BATCH_SIZE,1)
        z = torch.cat([z, c], 1)
        out3 = self.decoder_fc(z).reshape(self.batch_size,16,self.z_dim,self.z_dim,self.z_dim)
        return self.decoder_deconv(out3)
    

class Discriminator(torch.nn.Module):
    def __init__(self, z_dim, batch_size, latent, device):
        
        super(Discriminator, self).__init__()
        # channel_in = 128
        # H = 128
        # W = 128
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.latent = latent
        self.device = device
        
        self.conv_net = torch.nn.Sequential(

            # dim: Bx64x64x64
            torch.nn.Conv3d(in_channels=2, out_channels=16, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(16),
            torch.nn.LeakyReLU(0.2),
            
            # dim: Bx32x32x32
            torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(32),
            torch.nn.LeakyReLU(0.2),
      
            # dim: Bx16x16x16
            torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(64),
            torch.nn.LeakyReLU(0.2),

            # dim: Bx8x8x8
            torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            
            # dim: Bx4x4x4
            torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            
            # dim: Bx2x2x2
            torch.nn.Conv3d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=0),
        )
        
    def forward(self, z, c):
        # z dim: Bx128x128x128
        c = c.reshape(self.batch_size,1,1,1,1).repeat(1,1,self.z_dim*8,self.z_dim*8,self.z_dim*8)
        x = torch.cat([z, c], 1)
        return self.conv_net(x)

def gradient_penalty(critic, real, fake, c1, device="cpu"):
    """
    Gradient penalty for WGAN-GP
    Parameters
    ----------
    critic : :obj:`torch.nn.Module`
        Critic model of WGAN
    real : :obj:`torch.Tensor`
        Tensor of real data of size BxCxWxH
    fake : :obj:`torch.Tensor`
        Tensor of fake data of size BxCxWxH
    device : :obj:`str`
        Device to run the computation cpu or cuda
    Returns
    -------
    : :obj:`torch.Tensor`
        Scalar value of gradient penalty
    """
    BATCH_SIZE, C, H, W, D = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1, 1)).repeat(1, C, H, W, D).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, c1)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)

    return torch.mean((gradient_norm - 1) ** 2)
    # BATCH_SIZE, C, H, W, D = real.shape
    # c1 = c1.reshape(BATCH_SIZE, 1)
    # beta = torch.rand((BATCH_SIZE, 1, 1, 1, 1)).repeat(1, C, H, W, D).to(device)
    # beta2 = torch.rand((BATCH_SIZE, 1)).to(device)
    # interpolated_images = real * beta + fake.detach() * (1 - beta)
    # interpolated_c = c1 * beta2 + c2.detach() * (1 - beta2)
    # interpolated_images.requires_grad_(True)
    # interpolated_c.requires_grad_(True)

    # # Calculate critic scores
    # mixed_scores = critic(interpolated_images, interpolated_c)

    # # Take the gradient of the scores with respect to the images
    # gradient = torch.autograd.grad(
    #     inputs=(interpolated_images, interpolated_c),
    #     outputs=mixed_scores,
    #     grad_outputs=torch.ones_like(mixed_scores),
    #     create_graph=True,
    #     retain_graph=True,
    # )
    # gradients_x = gradient[0].view(gradient[0].size(0), -1)
    # gradients_c = gradient[1].view(gradient[1].size(0), -1)
    # gradient_penalty = ((gradients_x.norm(2, dim=1) - 1) ** 2).mean() + ((gradients_c.norm(2, dim=1) - 1) ** 2).mean()
    # return gradient_penalty
