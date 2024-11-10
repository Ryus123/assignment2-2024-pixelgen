import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import Generator, Discriminator, Calibrator_linear
from utils_MHGAN import D_train, G_train, Calibrator_train, save_models, save_model_C, load_model_G, load_model_D


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--load", type=bool, default=True,
                        help="Train only the calibrator and load checkpoint for G and D")

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
    # len(train_dataset) = 60000    
    # 10% of the training dataset is for the calibration
    # calibration_dataset = torch.utils.data.Subset(train_dataset,np.arange(54000, 60000, 1, dtype=int))
    # train_dataset = torch.utils.data.Subset(train_dataset,np.arange(0, 54000, 1, dtype=int))
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    # calibration_loader = torch.utils.data.DataLoader(dataset=calibration_dataset, 
    #                                            batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=True)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = Generator(g_output_dim = mnist_dim)
    D_tilde = Discriminator(mnist_dim)
    if args.load:
        G = load_model_G(G, 'checkpoints')
        D_tilde = load_model_D(D_tilde, 'checkpoints')
    G = torch.nn.DataParallel(G).cuda()
    D_tilde = torch.nn.DataParallel(D_tilde).cuda()
    C = torch.nn.DataParallel(Calibrator_linear()).cuda()

    # model = DataParallel(model).cuda()
    print('Model loaded.')

    # Optimizer 
    # define loss
    criterion = nn.BCELoss() 
    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_tilde_optimizer = optim.Adam(D_tilde.parameters(), lr = args.lr)
    C_optimizer = optim.Adam(C.parameters(), lr = args.lr)

    print('Start Training :')

    n_epoch = args.epochs
    if args.load == False:
        # We first train the Generator and Discriminator
        for epoch in trange(1, n_epoch+1, leave=True):           
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(-1, mnist_dim)
                D_train(x, G, D_tilde, D_tilde_optimizer, criterion)
                G_train(x, G, D_tilde, G_optimizer, criterion)

            if epoch % 10 == 0:
                save_models(G, D_tilde, 'checkpoints')

    # We then train the Calibrator
    for epoch in trange(1, n_epoch+1, leave=True):
        for batch_idx, (x,_) in enumerate(test_loader):
            x = x.view(-1, mnist_dim)
            Calibrator_train(x, C, G, D_tilde, C_optimizer, criterion)
        
        if epoch % 10 == 0:
            save_model_C(C, 'checkpoints')

                
    print('Training done')

        
