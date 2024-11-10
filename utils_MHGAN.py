import torch
import os
from torchvision import datasets, transforms

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    # D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()
    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return D_loss.data.item()


def Calibrator_train(x, C, G, D_tilde, C_optimizer, criterion):
    C.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    C_output = C(D_tilde(x_real))
    C_real_loss = criterion(C_output, y_real)

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    C_output = C(D_tilde(x_fake))
    C_fake_loss = criterion(C_output, y_fake)

    # gradient backprop & optimize the calibrator's parameters
    C_loss = C_real_loss + C_fake_loss
    C_loss.backward()
    C_optimizer.step()
    return C_loss.data.item()

def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()  # Latent space sample (input for G)
    y_real_labels = torch.ones(x.shape[0], 1).cuda()  # Real labels

    G_output = G(z)  # Generate fake samples
    D_output = D(G_output)  # Discriminator's evaluation of fake samples
    G_loss = criterion(D_output, y_real_labels)  # Generator's loss

    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()


def pick_up_real_samples(data_loader):
    """
    This function will return 1 real sample.
    """
    real_data = None
    for batch_idx, (x, _) in enumerate(data_loader):
        real_data = x
        break
    
    return real_data[:,0,:,:]


def mh_samples(x0, G, Discr, Calib, K, mnist_dim=784):
    """
    Return a new sample to save with the MH process.
    The output is of the shape (1, 28, 28)
    x0 shape = (2, mnist_dim)
    G: Generator
    Discr: Discriminator
    Calib: Calibrator
    K: Budget for length of MCMC
    """
    change = False
    max_ite = 0
    D_x0 =  Calib(Discr(x0))[0]
    
    while (change == False) and (max_ite <= 1):
        # Generate K fake:
        z = torch.randn(K, 100).cuda()
        x_pot = G(z)   # shape: (K, 784)
        # Get the Discrimninator's and x0's probability:
        D_output = Calib(Discr(x_pot))  # shape: (K)
        # Get the K uniform variable between [0;1]
        U = torch.rand(K).cuda()

        # MCMC process:
        for k in range(K):           
            ratio = (D_x0**(-1) - 1) / (D_output[k]**(-1) - 1)

            if U[k] <= ratio:
                change = True
                x0[0,:] = x_pot[k,:].clone()
                D_x0 =  Calib(Discr(x0))[0]

        # If change == True, we accepted a sample and we return it
        if change == True:
            return x0[0,:].reshape(1, 28, 28) 

        # Else we start a new chain with a fake image as the starting point
        else:
            x0 = G(torch.randn(2, 100).cuda())
            D_x0 = Calib(Discr(x0))[0]
            max_ite += 1

    # If we reach this part, then we have never accepted any samples.
    # We returned the first sample of the second chain.
    return x0[0,:].reshape(1, 28, 28)         

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))

def save_model_C(C, folder):
    torch.save(C.state_dict(), os.path.join(folder,'C.pth'))

def load_model_G(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def load_model_D(D, folder):
    ckpt = torch.load(os.path.join(folder,'D.pth'))
    D.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return D

def load_model_C(C, folder):
    ckpt = torch.load(os.path.join(folder,'C.pth'))
    C.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return C