import os
import random
from torch.utils.data import Dataset
from PIL import Image

from .augmentation import MotionAugmentation


class ImageMatteDataset(Dataset):
    def __init__(self,
                 imagematte_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform):
        self.imagematte_dir = imagematte_dir
        self.imagematte_files = os.listdir(os.path.join(imagematte_dir, 'fgr'))
        self.background_image_dir = background_image_dir
        self.background_image_files = os.listdir(background_image_dir)
        self.background_video_dir = background_video_dir
        self.background_video_clips = os.listdir(background_video_dir)
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.size = size
        self.transform = transform
        
    def __len__(self):
        return max(len(self.imagematte_files), len(self.background_image_files) + len(self.background_video_clips))
    
    def __getitem__(self, idx):
        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()
        
        fgrs, phas = self._get_imagematte(idx)
        
        if self.transform is not None:
            return self.transform(fgrs, phas, bgrs)
        
        return fgrs, phas, bgrs
    
    def _get_imagematte(self, idx):
        with Image.open(os.path.join(self.imagematte_dir, 'fgr', self.imagematte_files[idx % len(self.imagematte_files)])) as fgr, \
             Image.open(os.path.join(self.imagematte_dir, 'pha', self.imagematte_files[idx % len(self.imagematte_files)])) as pha:
            fgr = self._downsample_if_needed(fgr.convert('RGB'))
            pha = self._downsample_if_needed(pha.convert('L'))
        fgrs = [fgr] * self.seq_length
        phas = [pha] * self.seq_length
        return fgrs, phas
    
    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, self.background_image_files[random.choice(range(len(self.background_image_files)))])) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        bgrs = [bgr] * self.seq_length
        return bgrs

    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)
        return bgrs
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

class ImageMatteAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.95,
            prob_bgr_affine=0.3,
            prob_noise=0.05,
            prob_color_jitter=0.3,
            prob_grayscale=0.03,
            prob_sharpness=0.05,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )
