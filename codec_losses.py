import torch
import torchvision
from kornia import color
from color_transforms_255 import ycbcr_to_rgb_255, rgb_to_ycbcr_255, rgb_to_y_255
from pytorch_msssim import ms_ssim
from torch import nn
from torch.nn import functional as F
from pytorch_msssim import ssim
from math import ceil
# inputs of loss funcs are assumed to be in rgb with a range of [0,1] for all models except JPEGAI, 
# for JPEGAI RGB [0,255] for x, x_hat; YCbCr [0,255] for y, y_hat
loss_color_space = 'YCbCr'
def process_colorspace(x, x_hat, y, y_hat, is_jpegai):
    if loss_color_space == 'rgb':
        if is_jpegai:
            y = ycbcr_to_rgb_255(y)
            y_hat = ycbcr_to_rgb_255(y_hat)
        else:
            pass
    elif loss_color_space == 'YCbCr':
        if is_jpegai:
            print('jpegai ycbcr loss color conversion')
            x = rgb_to_ycbcr_255(x)
            x_hat = rgb_to_ycbcr_255(x_hat)
            print(f'x, x_hat, y, y_hat mins: {x.min()}, {x_hat.min()}, {y.min()}, {y_hat.min()}')
            print(f'x, x_hat, y, y_hat maxs: {x.max()}, {x_hat.max()}, {y.max()}, {y_hat.max()}')
        else:
            x = color.rgb_to_ycbcr(x)
            x_hat = color.rgb_to_ycbcr(x_hat)
            y = color.rgb_to_ycbcr(y)
            y_hat = color.rgb_to_ycbcr(y_hat)
    return x, x_hat, y, y_hat

class DifferentiableTextureDetector(nn.Module):
    def __init__(self, pool_size=32, window_size=11):
        """
        Differentiable version of texture distortion detection
        
        Args:
            pool_size (int): Size of the average pooling kernel for smoothing
            window_size (int): Size of the SSIM sliding window (must be odd)
        """
        super().__init__()
        self.pool_size = pool_size
        self.window_size = window_size
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2)
        
    def forward(self, I_orig, I_neural, I_trad):
        """
        Compute texture distortion detection in a differentiable way
        
        Args:
            I_orig (torch.Tensor): Original image [B,C,H,W] in range [0,1]
            I_neural (torch.Tensor): Neural-compressed image [B,C,H,W]
            I_trad (torch.Tensor): Traditional-compressed image [B,C,H,W]
            
        Returns:
            torch.Tensor: Heatmap difference [B,1,H,W]
            torch.Tensor: Confidence scores [B]
        """
        # Convert RGB to YUV and extract Y channel
        # y_orig = self._rgb_to_grayscale(I_orig)
        # y_neural = self._rgb_to_grayscale(I_neural)
        # y_trad = self._rgb_to_grayscale(I_trad)
        y_orig = I_orig[None, :]
        y_neural = I_neural[None, :]
        y_trad = I_trad[None, :]
        
        # Compute MS-SSIM maps (using multi-scale structural similarity)
        H_neural = self._compute_ssim_map(y_orig, y_neural)
        H_trad = self._compute_ssim_map(y_orig, y_trad)
        
        # Compute difference and smooth
        delta_H = self.pool(H_trad - H_neural)
        
        # For visualization, you can return the full heatmap
        return delta_H

    def forward_mode(self, I_orig, I_att, I_neural, I_trad):
        """
        Compute texture distortion detection in a differentiable way
        
        Args:
            I_orig (torch.Tensor): Original image [B,C,H,W] in range [0,1]
            I_neural (torch.Tensor): Neural-compressed image [B,C,H,W]
            I_trad (torch.Tensor): Traditional-compressed image [B,C,H,W]
            
        Returns:
            torch.Tensor: Heatmap difference [B,1,H,W]
            torch.Tensor: Confidence scores [B]
        """
        # Convert RGB to YUV and extract Y channel
        # y_orig = self._rgb_to_grayscale(I_orig)
        # y_neural = self._rgb_to_grayscale(I_neural)
        # y_trad = self._rgb_to_grayscale(I_trad)
        y_orig = I_orig[None, :]
        y_neural = I_neural[None, :]
        y_trad = I_trad[None, :]
        y_att = I_att[None, :]
        
        # Compute MS-SSIM maps (using multi-scale structural similarity)
        H_neural = self._compute_ssim_map(y_att, y_neural)
        H_trad = self._compute_ssim_map(y_orig, y_trad)
        
        # Compute difference and smooth
        delta_H = self.pool(H_trad - H_neural)
        
        # For visualization, you can return the full heatmap
        return delta_H
    
    def _rgb_to_grayscale(self, img):
        """Convert RGB to grayscale (Y channel)"""
        if img.shape[1] == 3:
            # Using standard RGB to grayscale conversion weights
            return (0.299 * img[:,0:1,:,:] + 
                    0.587 * img[:,1:2,:,:] + 
                    0.114 * img[:,2:3,:,:])
        return img
    
    def _compute_ssim_map(self, img1, img2):
        """
        Compute local SSIM similarity map using sliding window approach
        Uses single-scale SSIM to handle small patches
        """
        window_size = self.window_size
        pad = window_size // 2
        img1_pad = F.pad(img1, (pad, pad, pad, pad), mode='reflect')
        img2_pad = F.pad(img2, (pad, pad, pad, pad), mode='reflect')
        
        # Unfold both images into sliding windows
        unfold = nn.Unfold(window_size, stride=1)
        patches1 = unfold(img1_pad).view(1, window_size*window_size, -1)
        patches2 = unfold(img2_pad).view(1, window_size*window_size, -1)
        
        # Reshape for batch processing
        B, C, H, W = img1.shape
        patches1 = patches1.permute(1,0,2).reshape(-1, 1, window_size, window_size)
        patches2 = patches2.permute(1,0,2).reshape(-1, 1, window_size, window_size)
        
        # Compute SSIM for each patch
        ssim_vals = []
        for i in range(0, patches1.shape[0], 256):  # Process in chunks to save memory
            chunk1 = patches1[i:i+256]
            chunk2 = patches2[i:i+256]
            
            # Use single-scale SSIM for small patches
            ssim_vals.append(ssim(chunk1, chunk2, 
                              data_range=1.0, 
                              size_average=False,
                              win_size=min(window_size, 7)))  # Max win_size=7 for small patches
        
        ssim_map = torch.cat(ssim_vals).view(B, H, W)
        return ssim_map

    def detect_artifacts(self, I_orig, I_neural, I_trad):
        """
        Non-differentiable version that returns bounding boxes like the original paper
        """
        with torch.no_grad():
            delta_H, _ = self.forward(I_orig, I_neural, I_trad)
            delta_H = delta_H.squeeze().cpu().numpy()
            
            # Find artifact center (non-differentiable)
            max_val = delta_H.max()
            centers = np.argwhere(delta_H == max_val)
            center = centers[0] if len(centers) > 0 else (0, 0)
            
            return {
                'center': center,
                'confidence': float(max_val),
                'heatmap': delta_H
            }

    def detect_artifacts_mode(self, I_orig, I_att, I_neural, I_trad):
        """
        Non-differentiable version that returns bounding boxes like the original paper
        """
        with torch.no_grad():
            delta_H, _ = self.forward(I_orig, I_att, I_neural, I_trad)
            delta_H = delta_H.squeeze().cpu().numpy()
            
            # Find artifact center (non-differentiable)
            max_val = delta_H.max()
            centers = np.argwhere(delta_H == max_val)
            center = centers[0] if len(centers) > 0 else (0, 0)
            
            return {
                'center': center,
                'confidence': float(max_val),
                'heatmap': delta_H
            }

detect = DifferentiableTextureDetector()

def new_mode_map(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False): # x - before compression, y - after, hat - attacked
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    delta_map = detect.forward_mode(x[:,0,:,:], x_hat[:,0,:,:], y_hat[:,0,:,:], y[:,0,:,:])

    _, h, w = delta_map.shape
    loss = torch.zeros((ceil(h / 75), ceil(w / 75)))

    for i in range(0, h, 75):
        for j in range(0, w, 75):
            loss[i // 75, j // 75] = delta_map[0, i:i+75, j:j+75].max()

    return -loss.sum()

def added_noises_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False): # x - before compression, y - after, hat - attacked
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.square(y_hat - y - (x_hat - x)).mean()

def added_noises_loss_Y(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False): 
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.square(y_hat[:,0,:,:] - y[:,0,:,:] - (x_hat[:,0,:,:] - x[:,0,:,:])).mean()

def reconstr_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False):
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.nn.functional.mse_loss(x_hat, y_hat)

def reconstr_loss_Y(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False):
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.nn.functional.mse_loss(x_hat[:,0,:,:], y_hat[:,0,:,:])

def src_reconstr_loss_Y(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False): 
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.nn.functional.mse_loss(x[:,0,:,:], y_hat[:,0,:,:])

def ftda_default_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False): 
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.nn.functional.mse_loss(y, y_hat)

def ftda_default_loss_Y(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False): 
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.nn.functional.mse_loss(y[:,0,:,:], y_hat[:,0,:,:])

def ftda_msssim_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False):
    data_range = 1
    if is_jpegai:
        data_range = 255.0
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return ms_ssim(y, y_hat, data_range=data_range)

def reconstruction_msssim_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False):
    data_range = 1
    if is_jpegai:
        data_range = 255.0
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return ms_ssim(x_hat, y_hat, data_range=data_range)

def bpp_increase_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False):
    return 1 - bpp_loss

loss_func_name = 'added_noises_loss'
loss_name_2_func = {
    'added_noises_loss':added_noises_loss,
    'reconstr_loss':reconstr_loss,
    'ftda_default_loss':ftda_default_loss,
    'ftda_msssim_loss':ftda_msssim_loss,
    'reconstruction_msssim_loss':reconstruction_msssim_loss,
    'bpp_increase_loss':bpp_increase_loss,
    'added_noises_loss_Y':added_noises_loss_Y,
    'reconstr_loss_Y':reconstr_loss_Y,
    'src_reconstr_loss_Y':src_reconstr_loss_Y,
    'ftda_default_loss_Y':ftda_default_loss_Y
}
# experimental
def pointwise_added_noises_loss(x, x_hat, y, y_hat, h_pos=250, w_pos=250, kernel_size=19, sigma=3): # x - before compression, y - after, hat - attacked
    # construct a mask based on the position of the pixel: create aa gaussian mask (with center in h_pos, w_pos) and apply it to the loss
    # it should work for dimensions B x C x H x W
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat)
    mask = torch.zeros_like(x)
    mask[:, :, h_pos, w_pos] = 1
    blur_op = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    mask = blur_op(mask)
    
    #return torch.nn.functional.mse_loss(x_hat, y_hat)
    return -torch.square((y_hat - y - (x_hat - x)) * mask).mean()


loss_func = loss_name_2_func[loss_func_name]