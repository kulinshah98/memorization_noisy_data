# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
from torch_utils import ambient_utils
from training.sampler import edm_sampler

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).



@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data


    def __call__(self, net, images, labels=None, nature_noise=None,  
                 current_sigma=0.0, augment_pipe=None):

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        # sample a sigma in [current_sigma, sigma_T]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        n = torch.randn_like(y) * sigma
        noisy_input = y + n
        D_yn = net(noisy_input, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        
        return loss


@persistence.persistent_class
class AmbientHighNoise_EDMLowNoise_Loss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5) -> None:
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data


    def __call__(self, net, images, labels=None, nature_noise=None, 
                 current_sigma=0.0, augment_pipe=None):
        
        # Loss function that combines EDM loss for low noise scales and ambient loss for high noise scales
        # EDM loss: Used when sampled noise scale (sigma) is less than current_sigma
        # Ambient loss: Used when sampled noise scale (sigma) is greater than or equal to current_sigma
        #
        # Args:
        #   net: Neural network model
        #   images: Input images tensor
        #   labels: Optional class labels tensor
        #   nature_noise: Nature noise tensor to add to images
        #   current_sigma: Current noise scale in training
        #   augment_pipe: Optional data augmentation pipeline
        #
        # Returns:
        #   loss: Combined EDM and ambient loss tensor
        #
        # The function:
        # 1. Samples noise scale sigma from log-normal distribution
        # 2. For EDM loss (sigma < current_sigma):
        #    - Adds Gaussian noise scaled by sigma to clean images
        #    - Predicts denoised images directly
        # 3. For ambient loss (sigma >= current_sigma):
        #    - Adds nature_noise to get noisy target
        #    - Adds additional diffusion noise scaled by sqrt(sigma^2 - current_sigma^2)
        #    - Predicts denoised images and converts to noisy target prediction
        # 4. Combines predictions using flags and computes weighted MSE loss
        
        
        # Convert current_sigma to tensor with same shape as input images
        current_sigma = torch.Tensor([current_sigma]).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(net.device)
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        n = torch.randn_like(y) * sigma
        noisy_input_edm = y + n
        label_edm = y

        ## Input for ambient loss (used for high noise scale)
        y_nature = y + nature_noise
        sigma_clamped = torch.clamp(sigma, min=current_sigma)
        diffusion_noise = torch.randn_like(y) * torch.sqrt(sigma_clamped ** 2 - current_sigma ** 2)
        y_diffusion_noisy = y_nature + diffusion_noise
        label_ambient = y_nature

        ## Create inputs combining edm and ambient inputs
        edm_flg = (sigma < current_sigma)
        noisy_target_flg = (sigma >= current_sigma)
        net_inp = (noisy_input_edm *  edm_flg + y_diffusion_noisy * noisy_target_flg)
        x0_pred = net(net_inp, sigma, labels, augment_labels=augment_labels)
        D_noisy_target_yn = ambient_utils.from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, y_diffusion_noisy, sigma, current_sigma)

        ## Calculating the loss function
        loss_inp = D_noisy_target_yn * noisy_target_flg + x0_pred * edm_flg
        loss_labels = label_ambient * noisy_target_flg + label_edm * edm_flg
        loss = weight * ((loss_inp - loss_labels) ** 2)
        
        return loss