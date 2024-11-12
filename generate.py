import torch 
import torchvision
import os
import argparse
from torchvision import datasets, transforms
import time

from model import Generator, Discriminator, Calibrator_linear
from utils import load_model
from utils_MHGAN import pick_up_real_samples, mh_samples, load_model_C


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    #Classic : generate from G; MH : Generate with Metropolis Hasting rejection sampling
    parser.add_argument("--gen_method", type=str, default='classic',
                      help="Generation method with/out rejection sampling (classic; MH)")
    args = parser.parse_args()




    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(g_output_dim = mnist_dim).cuda()
    model = load_model(model, 'checkpoints', point="G.pth")
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    
    print('Model loaded.')


    print('Start Generating')
    os.makedirs('samples', exist_ok=True)


    # NORMAL GENERATION :
    if args.gen_method == 'classic':
        n_samples = 0
        with torch.no_grad():
            while n_samples<10000:
                z = torch.randn(args.batch_size, 100).cuda()
                x = model(z)
                x = x.reshape(args.batch_size, 28, 28)
                for k in range(x.shape[0]):
                    if n_samples<10000:
                        torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                        n_samples += 1


    # MHGAN GENERATION :
    if args.gen_method == 'MH':
        Discr = Discriminator(mnist_dim).cuda()
        Discr = load_model(Discr, 'checkpoints', point="D.pth")
        Discr = torch.nn.DataParallel(Discr).cuda()
        Discr.eval()
        
        Calib = Calibrator_linear().cuda()
        Calib = load_model_C(Calib, 'checkpoints')
        Calib = torch.nn.DataParallel(Calib).cuda()
        Calib.eval()

        # Loading dataset for MHGAN sampling:
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])
        test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=1, shuffle=True)
        
        n_samples = 0
        K = 10
        with torch.no_grad():
            # t_start = time.time()
            while n_samples<10000:
                x0 = torch.zeros((2, mnist_dim)).cuda()
                x_real = pick_up_real_samples(test_loader)
                x0[0,:] = x_real.view(-1, mnist_dim).reshape(1, mnist_dim).cuda()
                new_sample = mh_samples(x0, model, Discr, Calib, K, f"plot{n_samples}")
                torchvision.utils.save_image(new_sample, os.path.join('samples', f'{n_samples}.png'))
                n_samples += 1
    
    # building_time = time.time() - t_start
    # print(f'computation time = {building_time:.2f}s')