import torch
import os
from torchvision import datasets, transforms


def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1).cuda()
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()
    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def G_double_train(x, G, D, G_optimizer, criterion, threshold, max_attempts=10):
    G.zero_grad()
    
    best_G_output, best_D_output = None, None
    attempts = 0

    while attempts < max_attempts:
        z = torch.randn(x.shape[0], 100)  # Latent space sample
        G_output = G(z)  # Generate fake samples
        D_output = D(G_output)  # Discriminator's evaluation
        a_output = D_output/(1-D_output)

        # Check if the generated samples meet the threshold
        if torch.mean(a_output).item() >= threshold:
            best_G_output, best_D_output = G_output, D_output
            break
        elif best_D_output is None or torch.mean(a_output).item() > torch.mean(best_D_output).item():
            # Keep track of the best attempt
            best_G_output, best_D_output = G_output, D_output

        attempts += 1

    # Calculate generator loss on the best samples found
    y_real_labels = torch.ones(x.shape[0], 1)  # Real labels for generator loss
    G_loss = criterion(best_D_output, y_real_labels)  # Generator's loss

    # Backpropagation and optimization
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()


def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder, point='G.pth'):
    ckpt = torch.load(os.path.join(folder, point))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def pick_up_real_samples(batch_size, mnist_dim = 784):
    real_data = None
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, shuffle=True)
    
    for batch_idx, (x, _) in enumerate(test_loader):
        real_data = x
        break
    
    return real_data[:,0,:,:]