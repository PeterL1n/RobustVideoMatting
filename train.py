"""
# First update `train_config.py` to set paths to your dataset locations.

# You may want to change `--num-workers` according to your machine's memory.
# The default num-workers=8 may cause dataloader to exit unexpectedly when
# machine is out of memory.

# Stage 1
python train.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 15 \
    --learning-rate-backbone 0.0001 \
    --learning-rate-aspp 0.0002 \
    --learning-rate-decoder 0.0002 \
    --learning-rate-refiner 0 \
    --checkpoint-dir checkpoint/stage1 \
    --log-dir log/stage1 \
    --epoch-start 0 \
    --epoch-end 20

# Stage 2
python train.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 50 \
    --learning-rate-backbone 0.00005 \
    --learning-rate-aspp 0.0001 \
    --learning-rate-decoder 0.0001 \
    --learning-rate-refiner 0 \
    --checkpoint checkpoint/stage1/epoch-19.pth \
    --checkpoint-dir checkpoint/stage2 \
    --log-dir log/stage2 \
    --epoch-start 20 \
    --epoch-end 22
    
# Stage 3
python train.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 40 \
    --seq-length-hr 6 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00001 \
    --learning-rate-refiner 0.0002 \
    --checkpoint checkpoint/stage2/epoch-21.pth \
    --checkpoint-dir checkpoint/stage3 \
    --log-dir log/stage3 \
    --epoch-start 22 \
    --epoch-end 23

# Stage 4
python train.py \
    --model-variant mobilenetv3 \
    --dataset imagematte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 40 \
    --seq-length-hr 6 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00005 \
    --learning-rate-refiner 0.0002 \
    --checkpoint checkpoint/stage3/epoch-22.pth \
    --checkpoint-dir checkpoint/stage4 \
    --log-dir log/stage4 \
    --epoch-start 23 \
    --epoch-end 28
"""


import argparse
import torch
import random
import os
from torch import nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms.functional import center_crop
from tqdm import tqdm

from dataset.videomatte import (
    VideoMatteDataset,
    VideoMatteTrainAugmentation,
    VideoMatteValidAugmentation,
)
from dataset.imagematte import (
    ImageMatteDataset,
    ImageMatteAugmentation
)
from dataset.coco import (
    CocoPanopticDataset,
    CocoPanopticTrainAugmentation,
)
from dataset.spd import (
    SuperviselyPersonDataset
)
from dataset.youtubevis import (
    YouTubeVISDataset,
    YouTubeVISAugmentation
)
from dataset.augmentation import (
    TrainFrameSampler,
    ValidFrameSampler
)
from model import MattingNetwork
from train_config import DATA_PATHS
from train_loss import matting_loss, segmentation_loss


class Trainer:
    def __init__(self, rank, world_size):
        self.parse_args()
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        self.init_writer()
        self.train()
        self.cleanup()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Model
        parser.add_argument('--model-variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
        # Matting dataset
        parser.add_argument('--dataset', type=str, required=True, choices=['videomatte', 'imagematte'])
        # Learning rate
        parser.add_argument('--learning-rate-backbone', type=float, required=True)
        parser.add_argument('--learning-rate-aspp', type=float, required=True)
        parser.add_argument('--learning-rate-decoder', type=float, required=True)
        parser.add_argument('--learning-rate-refiner', type=float, required=True)
        # Training setting
        parser.add_argument('--train-hr', action='store_true')
        parser.add_argument('--resolution-lr', type=int, default=512)
        parser.add_argument('--resolution-hr', type=int, default=2048)
        parser.add_argument('--seq-length-lr', type=int, required=True)
        parser.add_argument('--seq-length-hr', type=int, default=6)
        parser.add_argument('--downsample-ratio', type=float, default=0.25)
        parser.add_argument('--batch-size-per-gpu', type=int, default=1)
        parser.add_argument('--num-workers', type=int, default=8)
        parser.add_argument('--epoch-start', type=int, default=0)
        parser.add_argument('--epoch-end', type=int, default=16)
        # Tensorboard logging
        parser.add_argument('--log-dir', type=str, required=True)
        parser.add_argument('--log-train-loss-interval', type=int, default=20)
        parser.add_argument('--log-train-images-interval', type=int, default=500)
        # Checkpoint loading and saving
        parser.add_argument('--checkpoint', type=str)
        parser.add_argument('--checkpoint-dir', type=str, required=True)
        parser.add_argument('--checkpoint-save-interval', type=int, default=500)
        # Distributed
        parser.add_argument('--distributed-addr', type=str, default='localhost')
        parser.add_argument('--distributed-port', type=str, default='12355')
        # Debugging
        parser.add_argument('--disable-progress-bar', action='store_true')
        parser.add_argument('--disable-validation', action='store_true')
        parser.add_argument('--disable-mixed-precision', action='store_true')
        self.args = parser.parse_args()
        
    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.log('Initializing distributed')
        os.environ['MASTER_ADDR'] = self.args.distributed_addr
        os.environ['MASTER_PORT'] = self.args.distributed_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    def init_datasets(self):
        self.log('Initializing matting datasets')
        size_hr = (self.args.resolution_hr, self.args.resolution_hr)
        size_lr = (self.args.resolution_lr, self.args.resolution_lr)
        
        # Matting datasets:
        if self.args.dataset == 'videomatte':
            self.dataset_lr_train = VideoMatteDataset(
                videomatte_dir=DATA_PATHS['videomatte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=DATA_PATHS['background_videos']['train'],
                size=self.args.resolution_lr,
                seq_length=self.args.seq_length_lr,
                seq_sampler=TrainFrameSampler(),
                transform=VideoMatteTrainAugmentation(size_lr))
            if self.args.train_hr:
                self.dataset_hr_train = VideoMatteDataset(
                    videomatte_dir=DATA_PATHS['videomatte']['train'],
                    background_image_dir=DATA_PATHS['background_images']['train'],
                    background_video_dir=DATA_PATHS['background_videos']['train'],
                    size=self.args.resolution_hr,
                    seq_length=self.args.seq_length_hr,
                    seq_sampler=TrainFrameSampler(),
                    transform=VideoMatteTrainAugmentation(size_hr))
            self.dataset_valid = VideoMatteDataset(
                videomatte_dir=DATA_PATHS['videomatte']['valid'],
                background_image_dir=DATA_PATHS['background_images']['valid'],
                background_video_dir=DATA_PATHS['background_videos']['valid'],
                size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
                seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
                seq_sampler=ValidFrameSampler(),
                transform=VideoMatteValidAugmentation(size_hr if self.args.train_hr else size_lr))
        else:
            self.dataset_lr_train = ImageMatteDataset(
                imagematte_dir=DATA_PATHS['imagematte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=DATA_PATHS['background_videos']['train'],
                size=self.args.resolution_lr,
                seq_length=self.args.seq_length_lr,
                seq_sampler=TrainFrameSampler(),
                transform=ImageMatteAugmentation(size_lr))
            if self.args.train_hr:
                self.dataset_hr_train = ImageMatteDataset(
                    imagematte_dir=DATA_PATHS['imagematte']['train'],
                    background_image_dir=DATA_PATHS['background_images']['train'],
                    background_video_dir=DATA_PATHS['background_videos']['train'],
                    size=self.args.resolution_hr,
                    seq_length=self.args.seq_length_hr,
                    seq_sampler=TrainFrameSampler(),
                    transform=ImageMatteAugmentation(size_hr))
            self.dataset_valid = ImageMatteDataset(
                imagematte_dir=DATA_PATHS['imagematte']['valid'],
                background_image_dir=DATA_PATHS['background_images']['valid'],
                background_video_dir=DATA_PATHS['background_videos']['valid'],
                size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
                seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
                seq_sampler=ValidFrameSampler(),
                transform=ImageMatteAugmentation(size_hr if self.args.train_hr else size_lr))
            
        # Matting dataloaders:
        self.datasampler_lr_train = DistributedSampler(
            dataset=self.dataset_lr_train,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)
        self.dataloader_lr_train = DataLoader(
            dataset=self.dataset_lr_train,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_lr_train,
            pin_memory=True)
        if self.args.train_hr:
            self.datasampler_hr_train = DistributedSampler(
                dataset=self.dataset_hr_train,
                rank=self.rank,
                num_replicas=self.world_size,
                shuffle=True)
            self.dataloader_hr_train = DataLoader(
                dataset=self.dataset_hr_train,
                batch_size=self.args.batch_size_per_gpu,
                num_workers=self.args.num_workers,
                sampler=self.datasampler_hr_train,
                pin_memory=True)
        self.dataloader_valid = DataLoader(
            dataset=self.dataset_valid,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            pin_memory=True)
        
        # Segementation datasets
        self.log('Initializing image segmentation datasets')
        self.dataset_seg_image = ConcatDataset([
            CocoPanopticDataset(
                imgdir=DATA_PATHS['coco_panoptic']['imgdir'],
                anndir=DATA_PATHS['coco_panoptic']['anndir'],
                annfile=DATA_PATHS['coco_panoptic']['annfile'],
                transform=CocoPanopticTrainAugmentation(size_lr)),
            SuperviselyPersonDataset(
                imgdir=DATA_PATHS['spd']['imgdir'],
                segdir=DATA_PATHS['spd']['segdir'],
                transform=CocoPanopticTrainAugmentation(size_lr))
        ])
        self.datasampler_seg_image = DistributedSampler(
            dataset=self.dataset_seg_image,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)
        self.dataloader_seg_image = DataLoader(
            dataset=self.dataset_seg_image,
            batch_size=self.args.batch_size_per_gpu * self.args.seq_length_lr,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_seg_image,
            pin_memory=True)
        
        self.log('Initializing video segmentation datasets')
        self.dataset_seg_video = YouTubeVISDataset(
            videodir=DATA_PATHS['youtubevis']['videodir'],
            annfile=DATA_PATHS['youtubevis']['annfile'],
            size=self.args.resolution_lr,
            seq_length=self.args.seq_length_lr,
            seq_sampler=TrainFrameSampler(speed=[1]),
            transform=YouTubeVISAugmentation(size_lr))
        self.datasampler_seg_video = DistributedSampler(
            dataset=self.dataset_seg_video,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)
        self.dataloader_seg_video = DataLoader(
            dataset=self.dataset_seg_video,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_seg_video,
            pin_memory=True)
        
    def init_model(self):
        self.log('Initializing model')
        self.model = MattingNetwork(self.args.model_variant, pretrained_backbone=True).to(self.rank)
        
        if self.args.checkpoint:
            self.log(f'Restoring from checkpoint: {self.args.checkpoint}')
            self.log(self.model.load_state_dict(
                torch.load(self.args.checkpoint, map_location=f'cuda:{self.rank}')))
            
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model_ddp = DDP(self.model, device_ids=[self.rank], broadcast_buffers=False, find_unused_parameters=True)
        self.optimizer = Adam([
            {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
            {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp},
            {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
            {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
        ])
        self.scaler = GradScaler()
        
    def init_writer(self):
        if self.rank == 0:
            self.log('Initializing writer')
            self.writer = SummaryWriter(self.args.log_dir)
        
    def train(self):
        for epoch in range(self.args.epoch_start, self.args.epoch_end):
            self.epoch = epoch
            self.step = epoch * len(self.dataloader_lr_train)
            
            if not self.args.disable_validation:
                self.validate()
            
            self.log(f'Training epoch: {epoch}')
            for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_lr_train, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                # Low resolution pass
                self.train_mat(true_fgr, true_pha, true_bgr, downsample_ratio=1, tag='lr')

                # High resolution pass
                if self.args.train_hr:
                    true_fgr, true_pha, true_bgr = self.load_next_mat_hr_sample()
                    self.train_mat(true_fgr, true_pha, true_bgr, downsample_ratio=self.args.downsample_ratio, tag='hr')
                
                # Segmentation pass
                if self.step % 2 == 0:
                    true_img, true_seg = self.load_next_seg_video_sample()
                    self.train_seg(true_img, true_seg, log_label='seg_video')
                else:
                    true_img, true_seg = self.load_next_seg_image_sample()
                    self.train_seg(true_img.unsqueeze(1), true_seg.unsqueeze(1), log_label='seg_image')
                    
                if self.step % self.args.checkpoint_save_interval == 0:
                    self.save()
                    
                self.step += 1
                
    def train_mat(self, true_fgr, true_pha, true_bgr, downsample_ratio, tag):
        true_fgr = true_fgr.to(self.rank, non_blocking=True)
        true_pha = true_pha.to(self.rank, non_blocking=True)
        true_bgr = true_bgr.to(self.rank, non_blocking=True)
        true_fgr, true_pha, true_bgr = self.random_crop(true_fgr, true_pha, true_bgr)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
        
        with autocast(enabled=not self.args.disable_mixed_precision):
            pred_fgr, pred_pha = self.model_ddp(true_src, downsample_ratio=downsample_ratio)[:2]
            loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)

        self.scaler.scale(loss['total']).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        if self.rank == 0 and self.step % self.args.log_train_loss_interval == 0:
            for loss_name, loss_value in loss.items():
                self.writer.add_scalar(f'train_{tag}_{loss_name}', loss_value, self.step)
            
        if self.rank == 0 and self.step % self.args.log_train_images_interval == 0:
            self.writer.add_image(f'train_{tag}_pred_fgr', make_grid(pred_fgr.flatten(0, 1), nrow=pred_fgr.size(1)), self.step)
            self.writer.add_image(f'train_{tag}_pred_pha', make_grid(pred_pha.flatten(0, 1), nrow=pred_pha.size(1)), self.step)
            self.writer.add_image(f'train_{tag}_true_fgr', make_grid(true_fgr.flatten(0, 1), nrow=true_fgr.size(1)), self.step)
            self.writer.add_image(f'train_{tag}_true_pha', make_grid(true_pha.flatten(0, 1), nrow=true_pha.size(1)), self.step)
            self.writer.add_image(f'train_{tag}_true_src', make_grid(true_src.flatten(0, 1), nrow=true_src.size(1)), self.step)
            
    def train_seg(self, true_img, true_seg, log_label):
        true_img = true_img.to(self.rank, non_blocking=True)
        true_seg = true_seg.to(self.rank, non_blocking=True)
        
        true_img, true_seg = self.random_crop(true_img, true_seg)
        
        with autocast(enabled=not self.args.disable_mixed_precision):
            pred_seg = self.model_ddp(true_img, segmentation_pass=True)[0]
            loss = segmentation_loss(pred_seg, true_seg)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        if self.rank == 0 and (self.step - self.step % 2) % self.args.log_train_loss_interval == 0:
            self.writer.add_scalar(f'{log_label}_loss', loss, self.step)
        
        if self.rank == 0 and (self.step - self.step % 2) % self.args.log_train_images_interval == 0:
            self.writer.add_image(f'{log_label}_pred_seg', make_grid(pred_seg.flatten(0, 1).float().sigmoid(), nrow=self.args.seq_length_lr), self.step)
            self.writer.add_image(f'{log_label}_true_seg', make_grid(true_seg.flatten(0, 1), nrow=self.args.seq_length_lr), self.step)
            self.writer.add_image(f'{log_label}_true_img', make_grid(true_img.flatten(0, 1), nrow=self.args.seq_length_lr), self.step)
    
    def load_next_mat_hr_sample(self):
        try:
            sample = next(self.dataiterator_mat_hr)
        except:
            self.datasampler_hr_train.set_epoch(self.datasampler_hr_train.epoch + 1)
            self.dataiterator_mat_hr = iter(self.dataloader_hr_train)
            sample = next(self.dataiterator_mat_hr)
        return sample
    
    def load_next_seg_video_sample(self):
        try:
            sample = next(self.dataiterator_seg_video)
        except:
            self.datasampler_seg_video.set_epoch(self.datasampler_seg_video.epoch + 1)
            self.dataiterator_seg_video = iter(self.dataloader_seg_video)
            sample = next(self.dataiterator_seg_video)
        return sample
    
    def load_next_seg_image_sample(self):
        try:
            sample = next(self.dataiterator_seg_image)
        except:
            self.datasampler_seg_image.set_epoch(self.datasampler_seg_image.epoch + 1)
            self.dataiterator_seg_image = iter(self.dataloader_seg_image)
            sample = next(self.dataiterator_seg_image)
        return sample
    
    def validate(self):
        if self.rank == 0:
            self.log(f'Validating at the start of epoch: {self.epoch}')
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            with torch.no_grad():
                with autocast(enabled=not self.args.disable_mixed_precision):
                    for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_valid, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                        true_fgr = true_fgr.to(self.rank, non_blocking=True)
                        true_pha = true_pha.to(self.rank, non_blocking=True)
                        true_bgr = true_bgr.to(self.rank, non_blocking=True)
                        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
                        batch_size = true_src.size(0)
                        pred_fgr, pred_pha = self.model(true_src)[:2]
                        total_loss += matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)['total'].item() * batch_size
                        total_count += batch_size
            avg_loss = total_loss / total_count
            self.log(f'Validation set average loss: {avg_loss}')
            self.writer.add_scalar('valid_loss', avg_loss, self.step)
            self.model_ddp.train()
        dist.barrier()
    
    def random_crop(self, *imgs):
        h, w = imgs[0].shape[-2:]
        w = random.choice(range(w // 2, w))
        h = random.choice(range(w // 2, h))
        results = []
        for img in imgs:
            B, T = img.shape[:2]
            img = img.flatten(0, 1)
            img = F.interpolate(img, (max(h, w), max(h, w)), mode='bilinear', align_corners=False)
            img = center_crop(img, (h, w))
            img = img.reshape(B, T, *img.shape[1:])
            results.append(img)
        return results
    
    def save(self):
        if self.rank == 0:
            os.makedirs(self.args.checkpoint_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint_dir, f'epoch-{self.epoch}.pth'))
            self.log('Model saved')
        dist.barrier()
        
    def cleanup(self):
        dist.destroy_process_group()
        
    def log(self, msg):
        print(f'[GPU{self.rank}] {msg}')
            
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        Trainer,
        nprocs=world_size,
        args=(world_size,),
        join=True)
