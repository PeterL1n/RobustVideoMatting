import os
import numpy as np
import random
import json
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image


class CocoPanopticDataset(Dataset):
    def __init__(self,
                 imgdir: str,
                 anndir: str,
                 annfile: str,
                 transform=None):
        with open(annfile) as f:
            self.data = json.load(f)['annotations']
            self.data = list(filter(lambda data: any(info['category_id'] == 1 for info in data['segments_info']), self.data))
        self.imgdir = imgdir
        self.anndir = anndir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        img = self._load_img(data)
        seg = self._load_seg(data)
        
        if self.transform is not None:
            img, seg = self.transform(img, seg)
            
        return img, seg

    def _load_img(self, data):
        with Image.open(os.path.join(self.imgdir, data['file_name'].replace('.png', '.jpg'))) as img:
            return img.convert('RGB')
    
    def _load_seg(self, data):
        with Image.open(os.path.join(self.anndir, data['file_name'])) as ann:
            ann.load()
            
        ann = np.array(ann, copy=False).astype(np.int32)
        ann = ann[:, :, 0] + 256 * ann[:, :, 1] + 256 * 256 * ann[:, :, 2]
        seg = np.zeros(ann.shape, np.uint8)
        
        for segments_info in data['segments_info']:
            if segments_info['category_id'] in [1, 27, 32]: # person, backpack, tie
                seg[ann == segments_info['id']] = 255
        
        return Image.fromarray(seg)
    

class CocoPanopticTrainAugmentation:
    def __init__(self, size):
        self.size = size
        self.jitter = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    
    def __call__(self, img, seg):
        # Affine
        params = transforms.RandomAffine.get_params(degrees=(-20, 20), translate=(0.1, 0.1),
                                                    scale_ranges=(1, 1), shears=(-10, 10), img_size=img.size)
        img = F.affine(img, *params, interpolation=F.InterpolationMode.BILINEAR)
        seg = F.affine(seg, *params, interpolation=F.InterpolationMode.NEAREST)
        
        # Resize
        params = transforms.RandomResizedCrop.get_params(img, scale=(0.5, 1), ratio=(0.7, 1.3))
        img = F.resized_crop(img, *params, self.size, interpolation=F.InterpolationMode.BILINEAR)
        seg = F.resized_crop(seg, *params, self.size, interpolation=F.InterpolationMode.NEAREST)
        
        # Horizontal flip
        if random.random() < 0.5:
            img = F.hflip(img)
            seg = F.hflip(seg)
        
        # Color jitter
        img = self.jitter(img)
        
        # To tensor
        img = F.to_tensor(img)
        seg = F.to_tensor(seg)
        
        return img, seg
    

class CocoPanopticValidAugmentation:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img, seg):
        # Resize
        params = transforms.RandomResizedCrop.get_params(img, scale=(1, 1), ratio=(1., 1.))
        img = F.resized_crop(img, *params, self.size, interpolation=F.InterpolationMode.BILINEAR)
        seg = F.resized_crop(seg, *params, self.size, interpolation=F.InterpolationMode.NEAREST)
        
        # To tensor
        img = F.to_tensor(img)
        seg = F.to_tensor(seg)
        
        return img, seg