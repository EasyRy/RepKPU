import torch
import torch.nn as nn
from models.decoder import Decoder, Decoder_o, Decoder_s
from models.encoder import PointTransformer, PointTransformer_o


class RepKPU(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.encoder = PointTransformer(cfgs)
        self.decoder = Decoder_s(cfgs) if cfgs.simple else Decoder(cfgs)
    
    def forward(self, pos):
        feat = self.encoder(pos)
        return self.decoder(pos, feat) 


class RepKPU_o(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.r = cfgs.up_rate
        self.dim = cfgs.out_dim
        self.encoder = PointTransformer_o(cfgs)
        self.neck = Decoder_o(cfgs)
        self.skip_mlp = nn.Sequential(
            nn.Conv1d(cfgs.out_dim + cfgs.trans_dim, cfgs.trans_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(cfgs.trans_dim, cfgs.trans_dim, 1),
            nn.ReLU(inplace=True)
        )
        self.offset_mlp = nn.Sequential(
            nn.Conv1d(cfgs.trans_dim, cfgs.trans_dim//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(cfgs.trans_dim//2, 3, 1)
        )

    def forward(self, pos):
        B, _, N = pos.shape
        feat = self.encoder(pos) # (b, d, n)
        disp_feat, reg_loss = self.neck(pos, feat)
        disp_feat = self.skip_mlp(torch.cat([disp_feat, feat.unsqueeze(-1).repeat(1,1,1,self.r).view(B, self.dim, -1)], dim=1))
        offset = torch.tanh(self.offset_mlp(disp_feat))
        xyz_repeat = pos.unsqueeze(-1).repeat(1,1,1,self.r).view(B, 3, -1) # (b, 3, n*r)
        upsampled_xyz = xyz_repeat + offset
        return upsampled_xyz, reg_loss

