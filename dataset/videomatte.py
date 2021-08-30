import os
import random
from torch.utils.data import Dataset
from PIL import Image

from .augmentation import MotionAugmentation


class VideoMatteDataset(Dataset):
    def __init__(self,
                 videomatte_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform=None):
        self.background_image_dir = background_image_dir
        self.background_image_files = os.listdir(background_image_dir)
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]
        
        self.videomatte_dir = videomatte_dir
        self.videomatte_clips = sorted(os.listdir(os.path.join(videomatte_dir, 'fgr')))
        self.videomatte_frames = [sorted(os.listdir(os.path.join(videomatte_dir, 'fgr', clip))) 
                                  for clip in self.videomatte_clips]
        self.videomatte_idx = [(clip_idx, frame_idx) 
                               for clip_idx in range(len(self.videomatte_clips)) 
                               for frame_idx in range(0, len(self.videomatte_frames[clip_idx]), seq_length)]
        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

    def __len__(self):
        return len(self.videomatte_idx)
    
    def __getitem__(self, idx):
        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()
        
        fgrs, phas = self._get_videomatte(idx)
        
        if self.transform is not None:
            return self.transform(fgrs, phas, bgrs)
        
        return fgrs, phas, bgrs
    
    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as bgr:
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
    
    def _get_videomatte(self, idx):
        clip_idx, frame_idx = self.videomatte_idx[idx]
        clip = self.videomatte_clips[clip_idx]
        frame_count = len(self.videomatte_frames[clip_idx])
        fgrs, phas = [], []
        for i in self.seq_sampler(self.seq_length):
            frame = self.videomatte_frames[clip_idx][(frame_idx + i) % frame_count]
            with Image.open(os.path.join(self.videomatte_dir, 'fgr', clip, frame)) as fgr, \
                 Image.open(os.path.join(self.videomatte_dir, 'pha', clip, frame)) as pha:
                    fgr = self._downsample_if_needed(fgr.convert('RGB'))
                    pha = self._downsample_if_needed(pha.convert('L'))
            fgrs.append(fgr)
            phas.append(pha)
        return fgrs, phas
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

class VideoMatteTrainAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.3,
            prob_bgr_affine=0.3,
            prob_noise=0.1,
            prob_color_jitter=0.3,
            prob_grayscale=0.02,
            prob_sharpness=0.1,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )

class VideoMatteValidAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0,
            prob_bgr_affine=0,
            prob_noise=0,
            prob_color_jitter=0,
            prob_grayscale=0,
            prob_sharpness=0,
            prob_blur=0,
            prob_hflip=0,
            prob_pause=0,
        )
