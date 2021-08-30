import os
from torch.utils.data import Dataset
from PIL import Image


class SuperviselyPersonDataset(Dataset):
    def __init__(self, imgdir, segdir, transform=None):
        self.img_dir = imgdir
        self.img_files = sorted(os.listdir(imgdir))
        self.seg_dir = segdir
        self.seg_files = sorted(os.listdir(segdir))
        assert len(self.img_files) == len(self.seg_files)
        self.transform = transform
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        with Image.open(os.path.join(self.img_dir, self.img_files[idx])) as img, \
             Image.open(os.path.join(self.seg_dir, self.seg_files[idx])) as seg:
            img = img.convert('RGB')
            seg = seg.convert('L')
        
        if self.transform is not None:
            img, seg = self.transform(img, seg)
            
        return img, seg
