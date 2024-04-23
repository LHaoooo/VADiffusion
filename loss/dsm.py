import torch
import torch.nn as nn
from functools import partial
from torch.distributions.gamma import Gamma
import torch.nn.functional as F
from einops import reduce


def anneal_dsm_score_estimation(scorenet, x, labels=None, loss_type='eps', hook=None, cond=None, cond_mask=None, gamma=False, L1=False, all_frames=False):
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    version = getattr(net, 'version', 'DDPM').upper()
    net_type = getattr(net, 'type') if isinstance(getattr(net, 'type'), str) else 'v1'

    if all_frames:
        x = torch.cat([x, cond], dim=1)
        print("x shape ", x.shape )
        cond = None

    # z, perturbed_x
    if version == "SMLD":
        sigmas = net.sigmas
        if labels is None:
            labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)
        used_sigmas = sigmas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        z = torch.randn_like(x)
        perturbed_x = x + used_sigmas * z
    elif version == "DDPM" or version == "DDIM" or version == "FPNDM":
        alphas = net.alphas
        if labels is None:
            labels = torch.randint(0, len(alphas), (x.shape[0],), device=x.device)
        used_alphas = alphas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        if gamma:
            used_k = net.k_cum[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            used_theta = net.theta_t[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            z = Gamma(used_k, 1 / used_theta).sample()
            z = (z - used_k*used_theta) / (1 - used_alphas).sqrt()
        else:
            z = torch.randn_like(x)
            # print("z shape",z.shape)  # 8,1,256,256
        perturbed_x = used_alphas.sqrt() * x + (1 - used_alphas).sqrt() * z
        # print('preturb x:',perturbed_x.shape)  # 8,1,256,256
    scorenet = partial(scorenet, cond=cond)
    pred_noise = scorenet(perturbed_x, labels, cond_mask=cond_mask)
    # print('pred noise',pred_noise.shape)  # 8,1,256,256

    # Loss
    if L1:
        def pow_(x):
            return x.abs()
    else:
        def pow_(x):
            return 1 / 2. * x.square()
    
    if hook is not None:
        hook(loss_eps, labels)

    if loss_type == 'eps':
        loss_eps = pow_((z - pred_noise).reshape(len(x), -1)).sum(dim=-1)
        return loss_eps.mean(dim=0)
    elif loss_type == 'x0':
        x0 = (1 / used_alphas.sqrt()) * (perturbed_x - (1 - used_alphas).sqrt() * pred_noise)
        # print('x0 ',x0.shape)  # 8,1,256,256
        intensity_loss = Intensity_Loss(l_num=2).to(x.device)
        loss_x0 = intensity_loss(x0, x)
        return loss_x0
    elif loss_type == 'x0+g':
        x0 = (1 / used_alphas.sqrt()) * (perturbed_x - (1 - used_alphas).sqrt() * pred_noise)
        grad_loss = Gradient_Loss(1, x.shape[1], device=x.device)
        intensity_loss = Intensity_Loss(l_num=2).to(x.device)
        loss_x0 = intensity_loss(x0, x)
        loss_G = grad_loss(x0, x)
        return loss_x0+loss_G
    elif loss_type == 'min_snr_gamma_eps':
        snr = used_alphas/(1-used_alphas)
        clipped_snr = snr.clone()
        loss_snr_eps = F.mse_loss(pred_noise,z,reduction='none')
        loss_snr_eps = reduce(loss_snr_eps,'b ... -> b','mean')
        return ((clipped_snr.clamp_(max = 5)/snr)*loss_snr_eps).mean()
    elif loss_type == 'min_snr_gamma_x0':
        x0 = (1 / used_alphas.sqrt()) * (perturbed_x - (1 - used_alphas).sqrt() * pred_noise)
        snr = used_alphas/(1-used_alphas)
        clipped_snr = snr.clone()
        loss_snr_x0 = F.mse_loss(x0,x,reduction='none')
        loss_snr_x0 = reduce(loss_snr_x0,'b ... -> b','mean')
        return (clipped_snr.clamp_(max = 5)*loss_snr_x0).mean()

class Intensity_Loss(nn.Module):
    def __init__(self, l_num):
        super(Intensity_Loss, self).__init__()
        self.l_num = l_num

    def forward(self, gen_frames, gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** self.l_num))


class Gradient_Loss(nn.Module):
    def __init__(self, alpha, channels, device):
        super(Gradient_Loss, self).__init__()
        self.alpha = alpha
        self.device = device
        filter = torch.FloatTensor([[-1., 1.]]).to(device)

        self.filter_x = filter.view(1, 1, 1, 2).repeat(1, channels, 1, 1)
        self.filter_y = filter.view(1, 1, 2, 1).repeat(1, channels, 1, 1)

    def forward(self, gen_frames, gt_frames):
        # pos=torch.from_numpy(np.identity(channels,dtype=np.float32))
        # neg=-1*pos
        # filter_x=torch.cat([neg,pos]).view(1,pos.shape[0],-1)
        # filter_y=torch.cat([pos.view(1,pos.shape[0],-1),neg.vew(1,neg.shape[0],-1)])
        gen_frames_x = nn.functional.pad(gen_frames, (1, 0, 0, 0))
        gen_frames_y = nn.functional.pad(gen_frames, (0, 0, 1, 0))
        gt_frames_x = nn.functional.pad(gt_frames, (1, 0, 0, 0))
        gt_frames_y = nn.functional.pad(gt_frames, (0, 0, 1, 0))

        gen_dx = nn.functional.conv2d(gen_frames_x, self.filter_x)
        gen_dy = nn.functional.conv2d(gen_frames_y, self.filter_y)
        gt_dx = nn.functional.conv2d(gt_frames_x, self.filter_x)
        gt_dy = nn.functional.conv2d(gt_frames_y, self.filter_y)

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x ** self.alpha + grad_diff_y ** self.alpha)