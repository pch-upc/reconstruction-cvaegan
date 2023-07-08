import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from cvaegan.conditional_architecture import *
from cvaegan.utils import *
import porespy as ps

DATASET = torch.from_numpy(np.load('./data/bentheimer1000.npy')).reshape(1000,1,128,128,128)
POROSITY = torch.from_numpy(np.load('./data/bentheimer_conditional.npy')).float()
print(DATASET.shape)
print(POROSITY.shape)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device('cuda')

#Specifiying folder location to save models per epoch
CHECKPOINT_GEN = "./checkpoints/generator/"
CHECKPOINT_CRITIC = "./checkpoints/critic/"

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
Z_DIM = 16
LATENT = 512 
NUM_EPOCHS = 1001
CRITIC_ITERATIONS = 4
GENERATOR_ITERATIONS = 1
LAMBDA_GP = 50

# initialize data loader
loader = DataLoader(MyLoader(DATASET, POROSITY), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

seed_everything(seed=3407)

# initialize generator and critic
encoder = Encoder(Z_DIM, BATCH_SIZE, LATENT, DEVICE).to(DEVICE)
encoder.train()

gen = Generator(Z_DIM, BATCH_SIZE, LATENT, DEVICE).to(DEVICE)
gen.train()

critic = Discriminator(Z_DIM, BATCH_SIZE, LATENT, DEVICE).to(DEVICE)
critic.train()

# initialize optimizerstride
opt_encoder = optim.Adam(encoder.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
scheduler_encoder = optim.lr_scheduler.CosineAnnealingLR(opt_encoder, 4 * NUM_EPOCHS * GENERATOR_ITERATIONS)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(opt_gen, 4 * NUM_EPOCHS * GENERATOR_ITERATIONS)

opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
scheduler_critic = optim.lr_scheduler.CosineAnnealingLR(opt_critic, 4 * NUM_EPOCHS * CRITIC_ITERATIONS)

# fixed noise for display
fixed_noise = noise = torch.randn(BATCH_SIZE,LATENT).to(DEVICE)

# Criterion for measuring porosity difference
criterion = torch.nn.L1Loss()

# Training
losses_encoder = []
losses_gen = []
losses_critic = []

for epoch in range(NUM_EPOCHS):
    batches = tqdm(loader)
    mean_loss_encoder = 0
    mean_loss_gen = 0
    mean_loss_critic = 0
    for batch_idx, real_cond in enumerate(batches):
        real = real_cond[0].float().unsqueeze(1).to(DEVICE)
        cur_batch_size = real.shape[0]
        c1 = real_cond[1].reshape(BATCH_SIZE,1).to(DEVICE)

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size,LATENT).to(DEVICE)
            fake = gen(noise, c1)

            critic_real = critic(real, c1).reshape(-1)
            critic_fake = critic(fake, c1).reshape(-1)

            gp = gradient_penalty(critic, real, fake, c1, device=DEVICE)
            loss_critic = torch.mean(critic_fake) - torch.mean(critic_real) + LAMBDA_GP * gp
                           
            critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()
            scheduler_critic.step()

            # mean critic loss
            mean_loss_critic += loss_critic.item()

        for _ in range(GENERATOR_ITERATIONS):
            # Update encoder network
            z,mean,logvar = encoder(real,c1)
            recon_data = gen(z,c1)

            # Update G network 
            noise = torch.randn(cur_batch_size,LATENT).to(DEVICE)
            noise_label = 0.3*torch.rand(cur_batch_size, 1).to(DEVICE)
            fake = gen(noise, noise_label)

            c3 = 1-torch.mean(torch.round(recon_data),dim=[2,3,4])

            gen_fake = critic(fake, noise_label).reshape(-1)

            loss_encoder = loss_function(recon_data,real,mean,logvar) + 1000 * criterion(c3,c1)
            loss_gen = - torch.mean(gen_fake)

            encoder.zero_grad()
            gen.zero_grad()
            loss_encoder.backward(retain_graph=True)
            loss_gen.backward()
            opt_encoder.step()
            opt_gen.step()
            scheduler_encoder.step()
            scheduler_gen.step()

            # mean vae loss
            mean_loss_encoder += loss_encoder.item()

            # mean generator loss
            mean_loss_gen += loss_gen.item()

        batches.set_postfix(
            epoch=epoch,
            encoder_loss=loss_encoder.item(),
            gen_loss=loss_gen.item(),
            critic_loss=loss_critic.item(),
        )
   
    if epoch % 5 == 0:
        fig, ax = plt.subplots(2,3, figsize=(14,8))
        rl = real[0][0].reshape(128,128,128).detach().cpu().numpy()
        fk2 = np.round(gen(fixed_noise, c1)[0][0].reshape(128,128,128).detach().cpu().numpy())
        ax[0][0].imshow(gen(fixed_noise, c1)[0,0,64,:,:].detach().cpu().numpy(), cmap='gray')
        ax[1][0].imshow(rl[64,:,:], cmap='gray')
        porreal = 1-np.mean(rl)
        porfake2 = 1-np.mean(fk2)
        ax[0][1].imshow(np.round(fk2[64,:,:]), cmap='gray')
        ax[1][1].imshow(np.round(rl[64,:,:]), cmap='gray')
        ax[1][1].set_title(f'real={porreal:.4f}')
        ax[0][1].set_title(f'fake={porfake2:.4f}')
        #Losses (generator and critic)
        ax[0][2].plot(losses_gen, 'b',label='Generator', linewidth=2)
        ax[0][2].plot(losses_critic, 'darkorange',label='Critic', linewidth=2)
        ax[0][2].plot(losses_encoder, 'g',label='VAE', linewidth=2)
        ax[0][2].legend()
        ax[0][2].set_xlabel('Epochs')
        ax[0][2].set_ylabel('Loss')

        plt.savefig(f'./Rock/GenVAE_bentheimer_{epoch}.png')


    # save losses at each epoch
    losses_gen.append(mean_loss_gen / (batch_idx * GENERATOR_ITERATIONS))
    losses_critic.append(mean_loss_critic / (batch_idx * CRITIC_ITERATIONS))
    losses_encoder.append(mean_loss_encoder / (batch_idx * GENERATOR_ITERATIONS))
    
    # Save checkpoints
    #Uncomment the following to save checkpoints while training
    if epoch % 5 == 0:
        save_checkpoint(gen, opt_gen, path=CHECKPOINT_GEN + f"generatorVAE_bentheimer_{epoch}.pt")
        save_checkpoint(critic, opt_critic, path=CHECKPOINT_CRITIC + f"critic_bentheimer_{epoch}.pt")
