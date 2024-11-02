import torch
import os
import sampling


def D_train(x, G, D, D_optimizer, criterion, K=640):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    # D_real_score = D_output

    # train discriminator on fake
    x_fake = torch.zeros(x.shape)

    # Sampling fake data with MH method
    for sample in range(x.shape[0]):
        K_samples = torch.randn(K, 100).cuda()
        K_samples = G(K_samples)
        D_output = D(K_samples)
        x_k = sampling.mh_sample(D_output)
        x_fake[sample] = x_k
    
    y_fake = torch.zeros(x.shape[0], 1).cuda()
    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    # D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return D_loss.data.item()


def Calibrator_train(x, D, G, D_tilde, D_optimizer, criterion):
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(D_tilde(x_real))
    D_real_loss = criterion(D_output, y_real)

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output = D(D_tilde(x_fake))
    D_fake_loss = criterion(D_output, y_fake)

    # gradient backprop & optimize the calibrator's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
    return D_loss.data.item()

def G_train(x, G, D, D_tilde, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()  # Latent space sample (input for G)
    y_real_labels = torch.ones(x.shape[0], 1).cuda()  # Real labels

    G_output = G(z)  # Generate fake samples
    D_output = D(D_tilde(G_output))  # Discriminator's evaluation of fake samples
    G_loss = criterion(D_output, y_real_labels)  # Generator's loss

    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()


def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
