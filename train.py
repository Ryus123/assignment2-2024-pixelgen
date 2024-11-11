import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


from model import Generator, Discriminator
from utils import D_train, G_train, save_models, compute_ck, loss_G_OBRS #, G_double_train, load_model

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()


    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 



    # define loss
    criterion = nn.BCELoss() 

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)

    print('Start Training :')
    # D_loss = None
    # G_loss = None
    
    n_epoch = args.epochs
    k=2.6
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
# ## ------------- Basical training
    #         D_loss = D_train(x, G, D, D_optimizer, criterion)
    #         G_loss = G_train(x, G, D, G_optimizer, criterion)
    #     print(f'Epoch : {epoch} | Generator loss : {G_loss:.4f} | Discriminator loss : {D_loss:.4f}')


# ## ------------- Implementing Tw/OBRS
            # Update Discriminator
            D_loss = D_train(x, G, D, D_optimizer, criterion)

            # Update ck
            z = torch.randn(x.shape[0], 100).cuda()
            G_output = G(z)
            ck, r_opt, M, aO = compute_ck(fake_samples=G_output, D=D, K=k)
            
            # Update Generator
            G_optimizer.zero_grad()
            loss_G = loss_G_OBRS(r_opt, aO, K=k)
            loss_G.backward()
            G_optimizer.step()
        print(f'Epoch : {epoch} | Generator loss : {loss_G.data.item():.4f} | Discriminator loss : {D_loss:.4f}')

#--------------------END--------------------------------------

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

    print('Training done')
        
# import matplotlib.pyplot as plt 

# plt.plot(range(1, len(D_loss) + 1), D_loss, marker='o', color='b', linestyle='-', label='Loss basical discriminator')
# plt.plot(range(1, len(G_loss) + 1), G_loss, marker='x', color='r', linestyle='--', label='Loss basical generator')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Basical GAN\'s with BCE losses')
# plt.legend()  # Ajoute une légende pour différencier les deux pertes
# plt.grid(True)

# #plt.savefig("plot/Basic_config_loss_comparison.png", format='png', dpi=300)
# plt.savefig("plot/hinge_loss_lr_2e-4_comparison.png", format='png', dpi=300)
# plt.show()