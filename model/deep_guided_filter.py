import torch
from torch import nn
from torch.nn import functional as F

"""
Adopted from <https://github.com/wuhuikai/DeepGuidedFilter/>
"""

class DeepGuidedFilterRefiner(nn.Module):
    def __init__(self, hid_channels=16):
        super().__init__()
        self.box_filter = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False, groups=4)
        self.box_filter.weight.data[...] = 1 / 9
        self.conv = nn.Sequential(
            nn.Conv2d(4 * 2 + hid_channels, hid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(hid_channels, 4, kernel_size=1, bias=True)
        )
        
    def forward_single_frame(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        fine_x = torch.cat([fine_src, fine_src.mean(1, keepdim=True)], dim=1)
        base_x = torch.cat([base_src, base_src.mean(1, keepdim=True)], dim=1)
        base_y = torch.cat([base_fgr, base_pha], dim=1)
        
        mean_x = self.box_filter(base_x)
        mean_y = self.box_filter(base_y)
        cov_xy = self.box_filter(base_x * base_y) - mean_x * mean_y
        var_x  = self.box_filter(base_x * base_x) - mean_x * mean_x
        
        A = self.conv(torch.cat([cov_xy, var_x, base_hid], dim=1))
        b = mean_y - A * mean_x
        
        H, W = fine_src.shape[2:]
        A = F.interpolate(A, (H, W), mode='bilinear', align_corners=False)
        b = F.interpolate(b, (H, W), mode='bilinear', align_corners=False)
        
        out = A * fine_x + b
        fgr, pha = out.split([3, 1], dim=1)
        return fgr, pha
    
    def forward_time_series(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        B, T = fine_src.shape[:2]
        fgr, pha = self.forward_single_frame(
            fine_src.flatten(0, 1),
            base_src.flatten(0, 1),
            base_fgr.flatten(0, 1),
            base_pha.flatten(0, 1),
            base_hid.flatten(0, 1))
        fgr = fgr.unflatten(0, (B, T))
        pha = pha.unflatten(0, (B, T))
        return fgr, pha
    
    def forward(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        if fine_src.ndim == 5:
            return self.forward_time_series(fine_src, base_src, base_fgr, base_pha, base_hid)
        else:
            return self.forward_single_frame(fine_src, base_src, base_fgr, base_pha, base_hid)
