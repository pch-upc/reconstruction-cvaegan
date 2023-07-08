import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from conditional_architecture import *
from utils import *
import os

# Specify GPU location
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device('cuda')

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
Z_DIM = 16
LATENT = 512 

# generator
gen = Generator(Z_DIM, BATCH_SIZE, LATENT, DEVICE).to(DEVICE)
# optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
load_checkpoint("./generator_bentheimer.pt", \
                    model=gen, optimizer=opt_gen, lr=1e-3)

seed_everything(seed=107)


samples_conditional=[]

logging_por = [
# 0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10
# 0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15
# 0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18,0.18
# 0.21,0.21,0.21,0.21,0.21,0.21,0.21,0.21,0.21,0.21
# 0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25
0.10, 0.15, 0.20, 0.25, 0.30
]
noise = torch.randn(BATCH_SIZE,LATENT).to(DEVICE)
for i in logging_por:
    # noise = torch.randn(BATCH_SIZE,LATENT).to(DEVICE)
    c1 = torch.tensor(i).to(DEVICE)
    fake_images = gen(noise, c1).detach().cpu().numpy().reshape(128,128,128)
    # samples_conditional.append(fake_images)
    samples_conditional.append(np.round(fake_images))    

samples_conditional = np.array(samples_conditional)
print (samples_conditional.shape)
fig = plt.figure(figsize=(15, 6))
columns = 5
rows = 1
pore_sum =[]
for i in range(columns*rows):
    volume = samples_conditional[i,:,:,:]
    img = samples_conditional[i,10,:,:]
    pore = 1-np.mean(volume)
    pore_sum.append(pore)
    print(pore)
 
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img,cmap = 'gray')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(f'Porosity={pore:.4f}', fontsize=15)
fig.tight_layout()
plt.show()

