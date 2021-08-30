"""
python generate_imagematte_with_background_video.py \
    --imagematte-dir ../matting-data/Distinctions/test \
    --background-dir ../matting-data/BackgroundVideos_mp4/test \
    --resolution 512 \
    --out-dir ../matting-data/evaluation/distinction_motion_sd/ \
    --random-seed 11
    
Seed:
    10 - distinction-static
    11 - distinction-motion
    12 - adobe-static
    13 - adobe-motion
    
"""

import argparse
import os
import pims
import numpy as np
import random
from multiprocessing import Pool
from PIL import Image
# from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from torchvision import transforms
from torchvision.transforms import functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--imagematte-dir', type=str, required=True)
parser.add_argument('--background-dir', type=str, required=True)
parser.add_argument('--num-samples', type=int, default=20)
parser.add_argument('--num-frames', type=int, default=100)
parser.add_argument('--resolution', type=int, required=True)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--random-seed', type=int)
parser.add_argument('--extension', type=str, default='.png')
args = parser.parse_args()
    
random.seed(args.random_seed)

imagematte_filenames = os.listdir(os.path.join(args.imagematte_dir, 'fgr'))
random.shuffle(imagematte_filenames)

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

random.shuffle(background_filenames)

def lerp(a, b, percentage):
    return a * (1 - percentage) + b * percentage

def motion_affine(*imgs):
    config = dict(degrees=(-10, 10), translate=(0.1, 0.1),
                  scale_ranges=(0.9, 1.1), shears=(-5, 5), img_size=imgs[0][0].size)
    angleA, (transXA, transYA), scaleA, (shearXA, shearYA) = transforms.RandomAffine.get_params(**config)
    angleB, (transXB, transYB), scaleB, (shearXB, shearYB) = transforms.RandomAffine.get_params(**config)

    T = len(imgs[0])
    variation_over_time = random.random()
    for t in range(T):
        percentage = (t / (T - 1)) * variation_over_time
        angle = lerp(angleA, angleB, percentage)
        transX = lerp(transXA, transXB, percentage)
        transY = lerp(transYA, transYB, percentage)
        scale = lerp(scaleA, scaleB, percentage)
        shearX = lerp(shearXA, shearXB, percentage)
        shearY = lerp(shearYA, shearYB, percentage)
        for img in imgs:
            img[t] = F.affine(img[t], angle, (transX, transY), scale, (shearX, shearY), F.InterpolationMode.BILINEAR)
    return imgs


def process(i):
    imagematte_filename = imagematte_filenames[i % len(imagematte_filenames)]
    background_filename = background_filenames[i % len(background_filenames)]
    
    bgrs = pims.PyAVVideoReader(os.path.join(args.background_dir, background_filename))
    
    out_path = os.path.join(args.out_dir, str(i).zfill(4))
    os.makedirs(os.path.join(out_path, 'fgr'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'pha'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'com'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'bgr'), exist_ok=True)

    with Image.open(os.path.join(args.imagematte_dir, 'fgr', imagematte_filename)) as fgr, \
         Image.open(os.path.join(args.imagematte_dir, 'pha', imagematte_filename)) as pha:
        fgr = fgr.convert('RGB')
        pha = pha.convert('L')
        
    fgrs = [fgr] * args.num_frames
    phas = [pha] * args.num_frames
    fgrs, phas = motion_affine(fgrs, phas)
    
    for t in range(args.num_frames):
        fgr = fgrs[t]
        pha = phas[t]
        
        w, h = fgr.size
        scale = args.resolution / max(h, w)
        w, h = int(w * scale), int(h * scale)
        
        fgr = fgr.resize((w, h))
        pha = pha.resize((w, h))
        
        if h < args.resolution:
            pt = (args.resolution - h) // 2
            pb = args.resolution - h - pt
        else:
            pt = 0
            pb = 0
            
        if w < args.resolution:
            pl = (args.resolution - w) // 2
            pr = args.resolution - w - pl
        else:
            pl = 0
            pr = 0
            
        fgr = F.pad(fgr, [pl, pt, pr, pb])
        pha = F.pad(pha, [pl, pt, pr, pb])
        
        if i // len(imagematte_filenames) % 2 == 1:
            fgr = fgr.transpose(Image.FLIP_LEFT_RIGHT)
            pha = pha.transpose(Image.FLIP_LEFT_RIGHT)
            
        fgr.save(os.path.join(out_path, 'fgr', str(t).zfill(4) + args.extension))
        pha.save(os.path.join(out_path, 'pha', str(t).zfill(4) + args.extension))
        
        bgr = Image.fromarray(bgrs[t]).convert('RGB')
        w, h = bgr.size
        scale = args.resolution / min(h, w)
        w, h = int(w * scale), int(h * scale)
        bgr = bgr.resize((w, h))
        bgr = F.center_crop(bgr, (args.resolution, args.resolution))
        bgr.save(os.path.join(out_path, 'bgr', str(t).zfill(4) + args.extension))
        
        pha = np.asarray(pha).astype(float)[:, :, None] / 255
        com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
        com.save(os.path.join(out_path, 'com', str(t).zfill(4) + args.extension))
        
if __name__ == '__main__':
    r = process_map(process, range(args.num_samples), max_workers=10)

