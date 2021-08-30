"""
python generate_videomatte_with_background_video.py \
    --videomatte-dir ../matting-data/VideoMatte240K_JPEG_HD/test \
    --background-dir ../matting-data/BackgroundVideos_mp4/test \
    --resize 512 288 \
    --out-dir ../matting-data/evaluation/vidematte_motion_sd/
"""

import argparse
import os
import pims
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--videomatte-dir', type=str, required=True)
parser.add_argument('--background-dir', type=str, required=True)
parser.add_argument('--num-samples', type=int, default=20)
parser.add_argument('--num-frames', type=int, default=100)
parser.add_argument('--resize', type=int, default=None, nargs=2)
parser.add_argument('--out-dir', type=str, required=True)
args = parser.parse_args()

# Hand selected a list of videos
background_filenames = [
    "0000.mp4",
    "0007.mp4",
    "0008.mp4",
    "0010.mp4",
    "0013.mp4",
    "0015.mp4",
    "0016.mp4",
    "0018.mp4",
    "0021.mp4",
    "0029.mp4",
    "0033.mp4",
    "0035.mp4",
    "0039.mp4",
    "0050.mp4",
    "0052.mp4",
    "0055.mp4",
    "0060.mp4",
    "0063.mp4",
    "0087.mp4",
    "0086.mp4",
    "0090.mp4",
    "0101.mp4",
    "0110.mp4",
    "0117.mp4",
    "0120.mp4",
    "0122.mp4",
    "0123.mp4",
    "0125.mp4",
    "0128.mp4",
    "0131.mp4",
    "0172.mp4",
    "0176.mp4",
    "0181.mp4",
    "0187.mp4",
    "0193.mp4",
    "0198.mp4",
    "0220.mp4",
    "0221.mp4",
    "0224.mp4",
    "0229.mp4",
    "0233.mp4",
    "0238.mp4",
    "0241.mp4",
    "0245.mp4",
    "0246.mp4"
]

random.seed(10)
    
videomatte_filenames = [(clipname, sorted(os.listdir(os.path.join(args.videomatte_dir, 'fgr', clipname)))) 
                        for clipname in sorted(os.listdir(os.path.join(args.videomatte_dir, 'fgr')))]

random.shuffle(background_filenames)

for i in range(args.num_samples):
    bgrs = pims.PyAVVideoReader(os.path.join(args.background_dir, background_filenames[i % len(background_filenames)]))
    clipname, framenames = videomatte_filenames[i % len(videomatte_filenames)]
    
    out_path = os.path.join(args.out_dir, str(i).zfill(4))
    os.makedirs(os.path.join(out_path, 'fgr'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'pha'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'com'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'bgr'), exist_ok=True)
    
    base_t = random.choice(range(len(framenames) - args.num_frames))
    
    for t in tqdm(range(args.num_frames), desc=str(i).zfill(4)):
        with Image.open(os.path.join(args.videomatte_dir, 'fgr', clipname, framenames[base_t + t])) as fgr, \
             Image.open(os.path.join(args.videomatte_dir, 'pha', clipname, framenames[base_t + t])) as pha:
            fgr = fgr.convert('RGB')
            pha = pha.convert('L')
            
            if args.resize is not None:
                fgr = fgr.resize(args.resize, Image.BILINEAR)
                pha = pha.resize(args.resize, Image.BILINEAR)
                
            
            if i // len(videomatte_filenames) % 2 == 1:
                fgr = fgr.transpose(Image.FLIP_LEFT_RIGHT)
                pha = pha.transpose(Image.FLIP_LEFT_RIGHT)
            
            fgr.save(os.path.join(out_path, 'fgr', str(t).zfill(4) + '.png'))
            pha.save(os.path.join(out_path, 'pha', str(t).zfill(4) + '.png'))
        
        bgr = Image.fromarray(bgrs[t])
        bgr = bgr.resize(fgr.size, Image.BILINEAR)
        bgr.save(os.path.join(out_path, 'bgr', str(t).zfill(4) + '.png'))
        
        pha = np.asarray(pha).astype(float)[:, :, None] / 255
        com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
        com.save(os.path.join(out_path, 'com', str(t).zfill(4) + '.png'))
