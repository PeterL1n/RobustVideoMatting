import torch
from torch.nn import functional as F

# --------------------------------------------------------------------------------- Train Loss


def matting_loss(pred_fgr, pred_pha, true_fgr, true_pha):
    """
    Args:
        pred_fgr: Shape(B, T, 3, H, W)
        pred_pha: Shape(B, T, 1, H, W)
        true_fgr: Shape(B, T, 3, H, W)
        true_pha: Shape(B, T, 1, H, W)
    """
    loss = dict()
    # Alpha losses
    loss['pha_l1'] = F.l1_loss(pred_pha, true_pha)
    loss['pha_laplacian'] = laplacian_loss(pred_pha.flatten(0, 1), true_pha.flatten(0, 1))
    loss['pha_coherence'] = F.mse_loss(pred_pha[:, 1:] - pred_pha[:, :-1],
                                       true_pha[:, 1:] - true_pha[:, :-1]) * 5
    # Foreground losses
    true_msk = true_pha.gt(0)
    pred_fgr = pred_fgr * true_msk
    true_fgr = true_fgr * true_msk
    loss['fgr_l1'] = F.l1_loss(pred_fgr, true_fgr)
    loss['fgr_coherence'] = F.mse_loss(pred_fgr[:, 1:] - pred_fgr[:, :-1],
                                       true_fgr[:, 1:] - true_fgr[:, :-1]) * 5
    # Total
    loss['total'] = loss['pha_l1'] + loss['pha_coherence'] + loss['pha_laplacian'] \
                  + loss['fgr_l1'] + loss['fgr_coherence']
    return loss

def segmentation_loss(pred_seg, true_seg):
    """
    Args:
        pred_seg: Shape(B, T, 1, H, W)
        true_seg: Shape(B, T, 1, H, W)
    """
    return F.binary_cross_entropy_with_logits(pred_seg, true_seg)


# ----------------------------------------------------------------------------- Laplacian Loss


def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]

