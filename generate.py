import torch 
import torchvision
import os
import argparse
from utils import pick_up_real_samples
# import time


from model import Generator, Discriminator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()




    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim = mnist_dim).cuda()
    model = load_model(model, 'checkpoints', point="G.pth")
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    
    Discr = Discriminator(mnist_dim).cuda()
    Discr = load_model(Discr, 'checkpoints', point="D.pth")
    Discr = torch.nn.DataParallel(Discr).cuda()
    Discr.eval()

    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    # n_samples = 0
    # with torch.no_grad():
    #     while n_samples<10000:
    #         z = torch.randn(args.batch_size, 100).cuda()
    #         x = model(z)
    #         x = x.reshape(args.batch_size, 28, 28)
    #         for k in range(x.shape[0]):
    #             if n_samples<10000:
    #                 torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
    #                 n_samples += 1


    n_samples = 0
    # t_start = time.time()

    with torch.no_grad():
        
        while n_samples<10000:
            x0 = pick_up_real_samples(1).cuda()
            x0 = x0.view(-1, mnist_dim)
            change = False # True when x0 actualized
            
            for _ in range(10):                
                z = torch.randn(1, 100).cuda()
                x_pot = model(z)
                U = torch.rand(1).cuda() # draw a uniform variable between [0;1]
                ratio = (Discr(x0)**(-1) - 1) / (Discr(x_pot)**(-1) - 1)

                if U <= ratio:
                    change = True
                    x0 = x_pot.clone()
                    
            if change == True:
                x_pot = x_pot.reshape(1, 28, 28)        
                torchvision.utils.save_image(x_pot[0], os.path.join('samples', f'{n_samples}.png'))         
                n_samples += 1
    #             print(f'Nombre de sample généré : {n_samples}')
    
    # building_time = time.time() - t_start
    # print(f'computation time = {building_time:.2f}s')

    
