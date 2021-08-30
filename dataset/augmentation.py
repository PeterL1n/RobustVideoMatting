import easing_functions as ef
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F


class MotionAugmentation:
    def __init__(self,
                 size,
                 prob_fgr_affine,
                 prob_bgr_affine,
                 prob_noise,
                 prob_color_jitter,
                 prob_grayscale,
                 prob_sharpness,
                 prob_blur,
                 prob_hflip,
                 prob_pause,
                 static_affine=True,
                 aspect_ratio_range=(0.9, 1.1)):
        self.size = size
        self.prob_fgr_affine = prob_fgr_affine
        self.prob_bgr_affine = prob_bgr_affine
        self.prob_noise = prob_noise
        self.prob_color_jitter = prob_color_jitter
        self.prob_grayscale = prob_grayscale
        self.prob_sharpness = prob_sharpness
        self.prob_blur = prob_blur
        self.prob_hflip = prob_hflip
        self.prob_pause = prob_pause
        self.static_affine = static_affine
        self.aspect_ratio_range = aspect_ratio_range
        
    def __call__(self, fgrs, phas, bgrs):
        # Foreground affine
        if random.random() < self.prob_fgr_affine:
            fgrs, phas = self._motion_affine(fgrs, phas)

        # Background affine
        if random.random() < self.prob_bgr_affine / 2:
            bgrs = self._motion_affine(bgrs)
        if random.random() < self.prob_bgr_affine / 2:
            fgrs, phas, bgrs = self._motion_affine(fgrs, phas, bgrs)
                
        # Still Affine
        if self.static_affine:
            fgrs, phas = self._static_affine(fgrs, phas, scale_ranges=(0.5, 1))
            bgrs = self._static_affine(bgrs, scale_ranges=(1, 1.5))
        
        # To tensor
        fgrs = torch.stack([F.to_tensor(fgr) for fgr in fgrs])
        phas = torch.stack([F.to_tensor(pha) for pha in phas])
        bgrs = torch.stack([F.to_tensor(bgr) for bgr in bgrs])
        
        # Resize
        params = transforms.RandomResizedCrop.get_params(fgrs, scale=(1, 1), ratio=self.aspect_ratio_range)
        fgrs = F.resized_crop(fgrs, *params, self.size, interpolation=F.InterpolationMode.BILINEAR)
        phas = F.resized_crop(phas, *params, self.size, interpolation=F.InterpolationMode.BILINEAR)
        params = transforms.RandomResizedCrop.get_params(bgrs, scale=(1, 1), ratio=self.aspect_ratio_range)
        bgrs = F.resized_crop(bgrs, *params, self.size, interpolation=F.InterpolationMode.BILINEAR)

        # Horizontal flip
        if random.random() < self.prob_hflip:
            fgrs = F.hflip(fgrs)
            phas = F.hflip(phas)
        if random.random() < self.prob_hflip:
            bgrs = F.hflip(bgrs)

        # Noise
        if random.random() < self.prob_noise:
            fgrs, bgrs = self._motion_noise(fgrs, bgrs)
        
        # Color jitter
        if random.random() < self.prob_color_jitter:
            fgrs = self._motion_color_jitter(fgrs)
        if random.random() < self.prob_color_jitter:
            bgrs = self._motion_color_jitter(bgrs)
            
        # Grayscale
        if random.random() < self.prob_grayscale:
            fgrs = F.rgb_to_grayscale(fgrs, num_output_channels=3).contiguous()
            bgrs = F.rgb_to_grayscale(bgrs, num_output_channels=3).contiguous()
            
        # Sharpen
        if random.random() < self.prob_sharpness:
            sharpness = random.random() * 8
            fgrs = F.adjust_sharpness(fgrs, sharpness)
            phas = F.adjust_sharpness(phas, sharpness)
            bgrs = F.adjust_sharpness(bgrs, sharpness)
        
        # Blur
        if random.random() < self.prob_blur / 3:
            fgrs, phas = self._motion_blur(fgrs, phas)
        if random.random() < self.prob_blur / 3:
            bgrs = self._motion_blur(bgrs)
        if random.random() < self.prob_blur / 3:
            fgrs, phas, bgrs = self._motion_blur(fgrs, phas, bgrs)

        # Pause
        if random.random() < self.prob_pause:
            fgrs, phas, bgrs = self._motion_pause(fgrs, phas, bgrs)
        
        return fgrs, phas, bgrs
    
    def _static_affine(self, *imgs, scale_ranges):
        params = transforms.RandomAffine.get_params(
            degrees=(-10, 10), translate=(0.1, 0.1), scale_ranges=scale_ranges,
            shears=(-5, 5), img_size=imgs[0][0].size)
        imgs = [[F.affine(t, *params, F.InterpolationMode.BILINEAR) for t in img] for img in imgs]
        return imgs if len(imgs) > 1 else imgs[0] 
    
    def _motion_affine(self, *imgs):
        config = dict(degrees=(-10, 10), translate=(0.1, 0.1),
                      scale_ranges=(0.9, 1.1), shears=(-5, 5), img_size=imgs[0][0].size)
        angleA, (transXA, transYA), scaleA, (shearXA, shearYA) = transforms.RandomAffine.get_params(**config)
        angleB, (transXB, transYB), scaleB, (shearXB, shearYB) = transforms.RandomAffine.get_params(**config)
        
        T = len(imgs[0])
        easing = random_easing_fn()
        for t in range(T):
            percentage = easing(t / (T - 1))
            angle = lerp(angleA, angleB, percentage)
            transX = lerp(transXA, transXB, percentage)
            transY = lerp(transYA, transYB, percentage)
            scale = lerp(scaleA, scaleB, percentage)
            shearX = lerp(shearXA, shearXB, percentage)
            shearY = lerp(shearYA, shearYB, percentage)
            for img in imgs:
                img[t] = F.affine(img[t], angle, (transX, transY), scale, (shearX, shearY), F.InterpolationMode.BILINEAR)
        return imgs if len(imgs) > 1 else imgs[0]
    
    def _motion_noise(self, *imgs):
        grain_size = random.random() * 3 + 1 # range 1 ~ 4
        monochrome = random.random() < 0.5
        for img in imgs:
            T, C, H, W = img.shape
            noise = torch.randn((T, 1 if monochrome else C, round(H / grain_size), round(W / grain_size)))
            noise.mul_(random.random() * 0.2 / grain_size)
            if grain_size != 1:
                noise = F.resize(noise, (H, W))
            img.add_(noise).clamp_(0, 1)
        return imgs if len(imgs) > 1 else imgs[0]
    
    def _motion_color_jitter(self, *imgs):
        brightnessA, brightnessB, contrastA, contrastB, saturationA, saturationB, hueA, hueB \
            = torch.randn(8).mul(0.1).tolist()
        strength = random.random() * 0.2
        easing = random_easing_fn()
        T = len(imgs[0])
        for t in range(T):
            percentage = easing(t / (T - 1)) * strength
            for img in imgs:
                img[t] = F.adjust_brightness(img[t], max(1 + lerp(brightnessA, brightnessB, percentage), 0.1))
                img[t] = F.adjust_contrast(img[t], max(1 + lerp(contrastA, contrastB, percentage), 0.1))
                img[t] = F.adjust_saturation(img[t], max(1 + lerp(brightnessA, brightnessB, percentage), 0.1))
                img[t] = F.adjust_hue(img[t], min(0.5, max(-0.5, lerp(hueA, hueB, percentage) * 0.1)))
        return imgs if len(imgs) > 1 else imgs[0]
    
    def _motion_blur(self, *imgs):
        blurA = random.random() * 10
        blurB = random.random() * 10

        T = len(imgs[0])
        easing = random_easing_fn()
        for t in range(T):
            percentage = easing(t / (T - 1))
            blur = max(lerp(blurA, blurB, percentage), 0)
            if blur != 0:
                kernel_size = int(blur * 2)
                if kernel_size % 2 == 0:
                    kernel_size += 1 # Make kernel_size odd
                for img in imgs:
                    img[t] = F.gaussian_blur(img[t], kernel_size, sigma=blur)
    
        return imgs if len(imgs) > 1 else imgs[0]
    
    def _motion_pause(self, *imgs):
        T = len(imgs[0])
        pause_frame = random.choice(range(T - 1))
        pause_length = random.choice(range(T - pause_frame))
        for img in imgs:
            img[pause_frame + 1 : pause_frame + pause_length] = img[pause_frame]
        return imgs if len(imgs) > 1 else imgs[0]
    

def lerp(a, b, percentage):
    return a * (1 - percentage) + b * percentage


def random_easing_fn():
    if random.random() < 0.2:
        return ef.LinearInOut()
    else:
        return random.choice([
            ef.BackEaseIn,
            ef.BackEaseOut,
            ef.BackEaseInOut,
            ef.BounceEaseIn,
            ef.BounceEaseOut,
            ef.BounceEaseInOut,
            ef.CircularEaseIn,
            ef.CircularEaseOut,
            ef.CircularEaseInOut,
            ef.CubicEaseIn,
            ef.CubicEaseOut,
            ef.CubicEaseInOut,
            ef.ExponentialEaseIn,
            ef.ExponentialEaseOut,
            ef.ExponentialEaseInOut,
            ef.ElasticEaseIn,
            ef.ElasticEaseOut,
            ef.ElasticEaseInOut,
            ef.QuadEaseIn,
            ef.QuadEaseOut,
            ef.QuadEaseInOut,
            ef.QuarticEaseIn,
            ef.QuarticEaseOut,
            ef.QuarticEaseInOut,
            ef.QuinticEaseIn,
            ef.QuinticEaseOut,
            ef.QuinticEaseInOut,
            ef.SineEaseIn,
            ef.SineEaseOut,
            ef.SineEaseInOut,
            Step,
        ])()

class Step: # Custom easing function for sudden change.
    def __call__(self, value):
        return 0 if value < 0.5 else 1


# ---------------------------- Frame Sampler ----------------------------


class TrainFrameSampler:
    def __init__(self, speed=[0.5, 1, 2, 3, 4, 5]):
        self.speed = speed
    
    def __call__(self, seq_length):
        frames = list(range(seq_length))
        
        # Speed up
        speed = random.choice(self.speed)
        frames = [int(f * speed) for f in frames]
        
        # Shift
        shift = random.choice(range(seq_length))
        frames = [f + shift for f in frames]
        
        # Reverse
        if random.random() < 0.5:
            frames = frames[::-1]

        return frames
    
class ValidFrameSampler:
    def __call__(self, seq_length):
        return range(seq_length)
