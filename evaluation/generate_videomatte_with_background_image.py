"""
python generate_videomatte_with_background_image.py \
    --videomatte-dir ../matting-data/VideoMatte240K_JPEG_HD/test \
    --background-dir ../matting-data/Backgrounds/valid \
    --num-samples 25 \
    --resize 512 288 \
    --out-dir ../matting-data/evaluation/vidematte_static_sd/
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
parser.add_argument('--extension', type=str, default='.png')
args = parser.parse_args()
    
random.seed(10)

videomatte_filenames = [(clipname, sorted(os.listdir(os.path.join(args.videomatte_dir, 'fgr', clipname)))) 
                        for clipname in sorted(os.listdir(os.path.join(args.videomatte_dir, 'fgr')))]

background_filenames = os.listdir(args.background_dir)
random.shuffle(background_filenames)

for i in range(args.num_samples):
    
    clipname, framenames = videomatte_filenames[i % len(videomatte_filenames)]
    
    out_path = os.path.join(args.out_dir, str(i).zfill(4))
    os.makedirs(os.path.join(out_path, 'fgr'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'pha'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'com'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'bgr'), exist_ok=True)
    
    with Image.open(os.path.join(args.background_dir, background_filenames[i])) as bgr:
        bgr = bgr.convert('RGB')

    
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
            
            fgr.save(os.path.join(out_path, 'fgr', str(t).zfill(4) + args.extension))
            pha.save(os.path.join(out_path, 'pha', str(t).zfill(4) + args.extension))
        
        if t == 0:
            bgr = bgr.resize(fgr.size, Image.BILINEAR)
            bgr.save(os.path.join(out_path, 'bgr', str(t).zfill(4) + args.extension))
        else:
            os.symlink(str(0).zfill(4) + args.extension, os.path.join(out_path, 'bgr', str(t).zfill(4) + args.extension))
        
        pha = np.asarray(pha).astype(float)[:, :, None] / 255
        com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
        com.save(os.path.join(out_path, 'com', str(t).zfill(4) + args.extension))
