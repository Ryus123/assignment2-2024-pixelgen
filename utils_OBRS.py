import torch


def GAN_opt_likelihood(samples, D):
    """For a sample X, compute and return the likelihood ration r_opt
    and the sup(r_opt) denotate as M"""
    score = D(samples) #return T_opt for each x in X
    r_opt = torch.exp(score-1) # Gradient of the conjugate of KL divergenec function
    M = torch.max(r_opt)
    return r_opt, M
    
def optimal_a_function(ck, M, r_opt):
    """Compute the optimal acceptance function"""
    optimal_a = r_opt*(ck/M)
    optimal_a = torch.minimum(optimal_a, torch.full_like(optimal_a, 1))
    return optimal_a 

def Loss_ck(ck, M, r_opt, N, K) :
    """Compute the loss of ck used on the dichotomy algorithm"""
    optimal_a = optimal_a_function(ck, M, r_opt)
    ck_loss = torch.sum(optimal_a) - N/K 
    return ck_loss, optimal_a   
    
def compute_ck(fake_samples, D, K, threshold=1e-4, warning=100):
    """Update ck with dichotomy algorithm"""
    count = 0
    r_opt, M = GAN_opt_likelihood(fake_samples, D)
    N = fake_samples.size(-1)
    c_min, c_max = 1e-10, 1e10
    ck = (c_min + c_max)/2
    # Compute the loss L(ck)
    ck_loss, aO = Loss_ck(ck, M, r_opt, N, K)
    while torch.abs(ck_loss) >= threshold:
        count +=1
        if ck_loss>= threshold:
            c_max = ck
        elif ck_loss < -threshold:
            c_min = ck
        # Update ck
        ck = (c_min + c_max)/2
        #Update the Loss L(ck)
        ck_loss, aO = Loss_ck(ck, M, r_opt, N, K)
        # Provide infinit loop
        if count>warning:
            break
        
    return ck, r_opt, M, aO

def loss_G_OBRS(r_opt, aO, K):
    """Compute the gradient of G"""
    KaO = K*aO
    divergence_f = r_opt*torch.log(r_opt/KaO)
    return torch.mean(divergence_f).cuda()