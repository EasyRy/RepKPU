import torch
import torch.nn as nn
import math
from einops import repeat
from models.utils import get_knn_pts, index_points
from models.kernel_points.kernel_utils import load_kernels
import torch.nn.functional as F

# RepKPoints Extraction Module, REM
class REM(nn.Module):
    def __init__(self, cfgs, is_deform = True, need_repkpoints = True):
        super().__init__()
        self.conv_radius = cfgs.conv_radius if is_deform else cfgs.conv_radius * cfgs.rigid_scale
        self.kernel_radius = cfgs.kernel_radius
        self.kernel_point_receptive_radius = cfgs.kernel_point_receptive_radius
        self.neighbor_limits = cfgs.neighbor_limits
        self.num_kernel_points = cfgs.num_kernel_points
        self.dim = cfgs.kp_dim
        self.inf = 1e6
        self.is_kp_bias = cfgs.is_kp_bias
        self.is_kp_bn = cfgs.is_kp_bn
        self.need_repkpoints = need_repkpoints
        self.is_deform = is_deform

        kernel_point = torch.Tensor(load_kernels(self.kernel_radius, self.num_kernel_points, 1, 3, 'center')).squeeze(0).permute(1,0).contiguous()  # 3, nkp
        self.kp_pos = nn.Parameter(kernel_point, requires_grad=False) #3, nkp
        
        # deformable kernel points
        if self.is_deform:
            self.conv_begin_deform = nn.Conv1d(cfgs.in_dim, self.dim, 1)
            self.kp_weight_deform = nn.Parameter(torch.zeros(size=(self.num_kernel_points, self.dim, self.dim)))
            if self.is_kp_bias:
                self.kp_bias_deform = nn.Parameter(torch.zeros(size=(self.dim,))) # d_h
            else:
                self.kp_bias_deform = None
            if self.is_kp_bn:
                self.kp_bn_deform = nn.BatchNorm1d(self.dim)
            else:
                self.kp_bn_deform = None
            self.act_deform = nn.ReLU(inplace=True)
            self.conv_end_deform = nn.Conv1d(self.dim, cfgs.in_dim, 1)
            self.act_out_deform = nn.ReLU(inplace=True)
            self.deform_conv = nn.Conv1d(cfgs.in_dim, 3 * self.num_kernel_points, 1)

        # kernel point representation
        self.conv_begin = nn.Conv1d(cfgs.in_dim, self.dim, 1)
        self.kp_weight = nn.Parameter(torch.zeros(size=(self.num_kernel_points, self.dim, self.dim))) 
        if self.is_kp_bias:
            self.kp_bias = nn.Parameter(torch.zeros(size=(self.dim,))) # d_h
        else:
            self.kp_bias = None
        if self.is_kp_bn:
            self.kp_bn = nn.BatchNorm1d(self.dim)
        else:
            self.kp_bn = None
        self.act = nn.ReLU(inplace=True)
        self.conv_end = nn.Conv1d(self.dim, self.dim, 1)
        self.act_out = nn.ReLU(inplace=True)

        if self.need_repkpoints:
            self.fuse_mlp = nn.Sequential(
                nn.Conv2d(self.dim*2, self.dim*2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.dim*2, cfgs.trans_dim, 1),
                nn.ReLU(inplace=True)
            )
        else:
            self.fuse_mlp = None

        self.reset_parameters()

    # deformable kernel points in RepKPoints
    def forward_deform(self, pos, feat):
        
        B, _, N = pos.shape
        pos_flipped = pos.permute(0, 2, 1) # B, N, 3
        neighbor_idx = self.query_ball_point(pos_flipped, pos_flipped, self.conv_radius) # (B, N, K)
            
        # kpconv
        neighbor_pos = self.index_points(pos, neighbor_idx, pad_inf=True).permute(0,3,1,2).contiguous() # (B, 3, N, K)
        
        if self.is_deform:
            neighbor_feat = self.index_points(self.conv_begin_deform(feat), neighbor_idx, pad_inf=False) # (B, N, K, d_in)
            neighbor_pos = neighbor_pos - pos.unsqueeze(-1)  # (B, 3, N, K) - (B, 3, N, 1) = (B, 3, N, K)
            differences = neighbor_pos.unsqueeze(-1) - self.kp_pos.view(1, 3, 1, 1, self.num_kernel_points) # (B, 3, N, K, 1) - (1, 3, 1, 1, nkp) = (B, 3, N, k, nkp)
            sq_distances = torch.sum(differences ** 2, dim=1)  # (B, N, k, nkp)
            neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.kernel_point_receptive_radius, min=0.0)  # (B, N, k, nkp)
            neighbor_weights = neighbor_weights.permute(0,1,3,2).contiguous() # (B, N, nkp, k)
            weighted_feats = torch.matmul(neighbor_weights, neighbor_feat)  # (B, N, nkp, k) x (B, N, K, d_in) = (B, N, nkp, d_in)
            output_feats = torch.einsum("bnkc,kcd->bnd", weighted_feats, self.kp_weight_deform) # B, N, d_h
            if self.kp_bias_deform != None:
                output_feats = output_feats + self.kp_bias_deform
            output_feats = output_feats.permute(0,2,1).contiguous()
            if self.kp_bn_deform != None:
                output_feats = self.kp_bn_deform(output_feats)
            output_feats = self.act_deform(output_feats)
            output_feats = self.act_out_deform(self.conv_end_deform(output_feats) + feat) # B, D, N
            offset = torch.tanh(self.deform_conv(output_feats)) # B, 3, nkp, N
            offset = offset.view(B, 3, -1, N).permute(0, 1, 3, 2).contiguous() # B, 3, N, nkp
            offset = offset * self.conv_radius
        else:
            offset = 0.0

        return offset, neighbor_pos, neighbor_idx

    def forward(self, pos, feat):
        deform_offset, rel_neighbor_pos, neighbor_idx = self.forward_deform(pos, feat) # (B, 3, N, nkp) (B, 3, N, k) (B, N, k)
        B, _, N = pos.shape
        neighbor_feat = self.index_points(self.conv_begin(feat), neighbor_idx, pad_inf=False) # (B, N, K, d_in)
        kp_pos_deform =  self.kp_pos.view(1, 3, 1, self.num_kernel_points) + deform_offset # (B, 3, N, nkp)
        differences = rel_neighbor_pos.unsqueeze(-1) - kp_pos_deform.unsqueeze(-2) # (B, 3, N, K, 1) - (B, 3, N, 1, nkp) = (B, 3, N, k, nkp)
        sq_distances = torch.sum(differences ** 2, dim=1)  # (B, N, k, nkp)

        if self.is_deform:
            reg_loss = self.fit_loss(sq_distances) + self.rep_loss(kp_pos_deform) 
        else:
            reg_loss = 0.0

        neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.kernel_point_receptive_radius, min=0.0)  # (B, N, k, nkp)
        neighbor_weights = neighbor_weights.permute(0,1,3,2).contiguous() # (B, N, nkp, k)
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feat)  # (B, N, nkp, k) x (B, N, K, d_in) = (B, N, nkp, d_in)
        output_feats = torch.einsum("bnkc,kcd->bnd", weighted_feats, self.kp_weight) # B, N, d_h
        if self.kp_bias != None:
            output_feats = output_feats + self.kp_bias
        output_feats = output_feats.permute(0,2,1).contiguous()
        if self.kp_bn != None:
            output_feats = self.kp_bn(output_feats)
        output_feats = self.act(output_feats)
        output_feats = self.act_out(self.conv_end(output_feats) + feat)

        if self.fuse_mlp != None:
            fuse_feats = [weighted_feats.permute(0, 3, 1, 2).contiguous(), output_feats.unsqueeze(-1).repeat(1,1,1,self.num_kernel_points)] #  (B, d*2, n, nkp)
            fuse_feats = self.fuse_mlp(torch.cat(fuse_feats, dim=1)) #  (B, d, n, nkp)
        else:
            fuse_feats = None
        
        # return [B, D, N]  [B, D, N, K]/None  [1]
        return output_feats, [kp_pos_deform, fuse_feats], reg_loss

    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm;
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def query_ball_point(self, xyz, new_xyz, radius):
        """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            xyz: all points, [B, N, 3]
            new_xyz: query points, [B, S, 3]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = self.square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :self.neighbor_limits]
        return group_idx

    def index_points(self, points, idx, pad_inf=False):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        points = points.permute(0,2,1).contiguous()
        B, N, C = points.shape
        device = points.device
        if pad_inf:
            shadow_points= torch.zeros(B, 1, C) + self.inf 
        else:
            shadow_points= torch.zeros(B, 1, C)
        shadow_points = shadow_points.to(device)
        cat_points = torch.cat([points, shadow_points], dim=1) # B, N+1, C

        B = cat_points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = cat_points[batch_indices, idx, :]
        return new_points

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.kp_weight, a=math.sqrt(5))
        if self.is_deform:
            nn.init.kaiming_uniform_(self.kp_weight_deform, a=math.sqrt(5))
        
        if self.kp_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kp_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.kp_bias, -bound, bound)
        if (self.is_deform) and (self.kp_bias_deform is not None):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kp_weight_deform)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.kp_bias_deform, -bound, bound)
        
    def fit_loss(self, sq_dis):
        # sq_dis  (B, N, k, nkp)
        sq_dis = torch.sqrt(sq_dis)
        sq_dis, _ = torch.min(sq_dis, dim=-2) # (B, N, nkp)
        loss = (sq_dis / self.conv_radius)**2 # (B, N, nkp)
        _l1 = nn.L1Loss()
        loss = _l1(loss, torch.zeros_like(loss).to(loss.device))
        return loss

    def rep_loss(self, deform_kp_pos):
        B, _, N, Nkp = deform_kp_pos.shape
        repulse_extent = self.kernel_point_receptive_radius 
        norm_kp_pos = deform_kp_pos / self.conv_radius #(B, 3, N, nkp) normalize
        norm_kp_pos = norm_kp_pos.permute(0,2,3,1).contiguous().view(-1, Nkp, 3) # (B*N, nkp, 3)
        loss = 0.0
        _l1 = nn.L1Loss()
        for i in range(self.num_kernel_points):
            other_KP = torch.cat([norm_kp_pos[:, :i, :], norm_kp_pos[:, i + 1:, :]], dim=1).detach() # (B*N, nkp-1, 3)
            distances = torch.sqrt(torch.sum((other_KP - norm_kp_pos[:, i:i + 1, :]) ** 2, dim=2)) # (B*N, nkp-1)
            rep_loss = torch.sum(torch.clamp_max(distances - repulse_extent, max=0.0) ** 2, dim=1)
            loss += _l1(rep_loss, torch.zeros_like(rep_loss)) / self.num_kernel_points
        return loss

# KP-Queries Generation Module, KGM
class KGM(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.conv_radius = cfgs.conv_radius * cfgs.rigid_scale
        self.kernel_radius = cfgs.kernel_radius * cfgs.query_scale
        self.kernel_point_receptive_radius = cfgs.kernel_point_receptive_radius * cfgs.query_scale
        self.neighbor_limits = cfgs.neighbor_limits
        self.dim = cfgs.kp_dim
        self.inf = 1e6
        self.r = cfgs.up_rate
        self.is_kp_bias = cfgs.is_kp_bias
        self.is_kp_bn = cfgs.is_kp_bn

        self.query_pos = nn.Parameter(torch.Tensor(load_kernels(self.kernel_radius, self.r+1, 1, 3, 'center')).squeeze(0).permute(1,0).contiguous().view(1,3,1,self.r+1), requires_grad=False)  # 1, 3, 1, r+1
        self.conv_begin_query = nn.Conv1d(self.dim, self.dim, 1)
        self.kp_weight_query = nn.Parameter(torch.zeros(size=(self.r+1, self.dim, self.dim))) # nkp, d_in, d_h
        if self.is_kp_bias:
            self.kp_bias_query = nn.Parameter(torch.zeros(size=(self.dim,))) # d_h
        else:
            self.kp_bias_query = None
        if self.is_kp_bn:
            self.bn_query = nn.BatchNorm1d(self.dim)
        else:
            self.bn_query = None
        self.act_query = nn.ReLU(inplace=True)
        self.conv_end_query = nn.Conv1d(self.dim, self.dim, 1)
        self.act_out_query = nn.ReLU(inplace=True)
        self.query_mlp = nn.Sequential(
            nn.Conv2d(self.dim*3+3, self.dim*2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim*2, cfgs.trans_dim, 1),
            nn.ReLU(inplace=True)
        )
        self.reset_parameters()
        
    def forward(self, pos, feat):
        B, _, N = pos.shape
        pos_flipped = pos.permute(0,2,1).contiguous() # B, N, 3
        neighbor_idx = self.query_ball_point(pos_flipped, pos_flipped, self.conv_radius) # (B, N, K)
        neighbor_pos = self.index_points(pos, neighbor_idx, pad_inf=True).permute(0,3,1,2).contiguous() # (B, 3, N, K)
        neighbor_pos = neighbor_pos - pos.unsqueeze(-1)  # (B, 3, N, K) - (B, 3, N, 1) = (B, 3, N, K)
        differences = neighbor_pos.unsqueeze(-1) - self.query_pos.view(1, 3, 1, 1, self.r+1) # (B, 3, N, K, 1) - (1, 3, 1, 1, r) = (B, 3, N, k, r)
        sq_distances = torch.sum(differences ** 2, dim=1)  # (B, N, k, r)
        neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / (self.kernel_point_receptive_radius), min=0.0)  # (B, N, k, r)
        neighbor_weights = neighbor_weights.permute(0,1,3,2).contiguous() # (B, N, r, k)
        neighbor_feat = self.index_points(self.conv_begin_query(feat), neighbor_idx, pad_inf=False) # (B, N, K, d_in)
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feat)  # (B, N, r, k) x (B, N, K, d_in) = (B, N, r, d_in) -> (B, d_in, N, r)
        
        query_feats = torch.einsum("bnkc,kcd->bnd", weighted_feats, self.kp_weight_query) # B, N, d_h
        if self.kp_bias_query != None:
            query_feats = query_feats + self.kp_bias_query
        query_feats = query_feats.permute(0,2,1).contiguous()
        if self.bn_query != None:
            query_feats = self.bn_query(query_feats)
        query_feats = self.act_query(query_feats)
        query_feats = self.act_out_query(self.conv_end_query(query_feats) + feat)
        queries = self.query_mlp(torch.cat([weighted_feats.permute(0, 3, 1, 2).contiguous()[:, :, :, 1:], self.query_pos.repeat(B, 1, N, 1)[:, :, :, 1:], \
            query_feats.unsqueeze(-1).repeat(1,1,1,self.r), feat.unsqueeze(-1).repeat(1,1,1,self.r)], dim=1)) # b, d, n, r

        # queries = self.query_mlp(torch.cat([weighted_feats.permute(0, 3, 1, 2).contiguous()[:, :, :, 1:], feat.unsqueeze(-1).repeat(1,1,1,self.r)], dim=1)) # b, d, n, r


        return [self.query_pos.squeeze()[:, 1:], queries]

    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm;
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def query_ball_point(self, xyz, new_xyz, radius):
        """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            xyz: all points, [B, N, 3]
            new_xyz: query points, [B, S, 3]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = self.square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :self.neighbor_limits]
        return group_idx

    def index_points(self, points, idx, pad_inf=False):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        points = points.permute(0,2,1).contiguous()
        B, N, C = points.shape
        device = points.device
        if pad_inf:
            shadow_points= torch.zeros(B, 1, C) + self.inf 
        else:
            shadow_points= torch.zeros(B, 1, C)
        shadow_points = shadow_points.to(device)
        cat_points = torch.cat([points, shadow_points], dim=1) # B, N+1, C

        B = cat_points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = cat_points[batch_indices, idx, :]
        return new_points

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.kp_weight_query, a=math.sqrt(5))
        if self.kp_bias_query is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kp_weight_query)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.kp_bias_query, -bound, bound) 




# Multi-head Attention
class MHA(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.dim = cfgs.trans_dim
        self.h = cfgs.head_num
        self.num_kernel_points = cfgs.num_kernel_points
        self.r = cfgs.up_rate
        self.conv_queries = nn.Conv2d(self.dim, self.dim, 1)
        self.conv_keys = nn.Conv2d(self.dim, self.dim, 1)
        self.conv_values = nn.Conv2d(self.dim, self.dim, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(self.dim, self.dim*2, 1),
            nn.BatchNorm2d(self.dim*2) if cfgs.is_attn_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim*2, self.dim, 1)
        )
        # self.attn_conv = nn.Conv2d(self.dim, self.dim, 1)
        self.ffn_act = nn.ReLU(inplace=True)

    def forward(self, queries, feats):
        B, _, N, __ = feats.shape
        queries_iden = queries
        queries = self.conv_queries(queries).view(B, self.h, self.dim//self.h, N, -1).permute(0, 3, 1, 4, 2).contiguous() # (B, n, h, r, d//h)
        keys = self.conv_keys(feats).view(B, self.h, self.dim//self.h, N, -1).permute(0, 3, 1, 2, 4).contiguous() # (B, n, h, d//h, nkp)
        values = self.conv_values(feats).view(B, self.h, self.dim//self.h, N, -1).permute(0, 3, 1, 4, 2).contiguous() # (B, n, h, nkp, d//h) 

        logits = torch.matmul(queries, keys) * ((self.dim // self.h)**(-0.5)) # (B, n, h, r, nkp) 
        soft_logits = torch.softmax(logits, dim=-1)
        agg_feats = torch.matmul(soft_logits, values).permute(0, 1, 2, 4, 3).contiguous().view(B, N, -1, self.r) # (B, n, d, r) 
        agg_feats = agg_feats.permute(0,2,1,3).contiguous()# (B, n, d, r)
        # agg_feats = self.attn_conv(agg_feats) + queries_iden# (B, d, N, r) 
        agg_feats = agg_feats + queries_iden
        agg_feats = self.ffn_act(self.ffn(agg_feats) + agg_feats)
        return agg_feats

# Cross-Attention Transformer
class Cross_Attn(nn.Module):
    def __init__(self, cfgs):
        super().__init__()    
        self.attns = nn.ModuleList()
        for i in range(cfgs.trans_num):
            self.attns.append(MHA(cfgs))

    def forward(self, q, k):
        for m in self.attns:
            q = m(q, k)
        return q

# Decoder
class Decoder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.r = cfgs.up_rate
        self.disp_dim = cfgs.trans_dim
        self.rem0 = REM(cfgs, is_deform = True, need_repkpoints = False)
        self.rem1 = REM(cfgs, is_deform = True, need_repkpoints = False)
        self.rem2 = REM(cfgs, is_deform = True, need_repkpoints = True)
        self.kgm = KGM(cfgs)
        self.cross_attn = Cross_Attn(cfgs)
        self.disp_mlp = nn.Sequential(
            nn.Conv1d(cfgs.trans_dim, 16, 1),  
            nn.ReLU(inplace = True),
            nn.Conv1d(16, 3, 1)
        )
        
        
        

    def forward(self, pos, feat):
        B, _, N = pos.shape
        local_feat_0, _, reg_loss0 = self.rem0(pos, feat)
        local_feat_1, _, reg_loss1 = self.rem1(pos, local_feat_0)
        local_feat_2, RepKPoints, reg_loss2 = self.rem2(pos, local_feat_1)
        KPQueries = self.kgm(pos, local_feat_2)
        disp_feat = self.cross_attn(KPQueries[1], RepKPoints[1]).view(B, self.disp_dim, -1) # b, d, n * r
        disp = torch.tanh(self.disp_mlp(disp_feat))
        new_pos = pos.unsqueeze(-1).repeat(1,1,1,self.r).view(B, 3, -1) + disp # b, 3, n*r

        return new_pos, reg_loss1 + reg_loss0 + reg_loss2


# Decoder simple
class MHA_o(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.dim = cfgs.trans_dim
        self.h = cfgs.head_num
        self.num_kernel_points = cfgs.num_kernel_points
        self.r = cfgs.up_rate
        self.conv_querys = nn.Conv2d(self.dim, self.dim, 1)
        self.conv_keys = nn.Conv2d(self.dim, self.dim, 1)
        self.conv_values = nn.Conv2d(self.dim, self.dim, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(self.dim, self.dim*2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim*2, self.dim, 1),
        )
        self.attn_conv = nn.Conv2d(self.dim, self.dim, 1)
        self.ffn_act = nn.ReLU(inplace=True)

    def forward(self, querys, feats):
        B, _, N, __ = feats.shape
        querys_iden = querys
        querys = self.conv_querys(querys).view(B, self.h, self.dim//self.h, N, -1).permute(0, 3, 1, 4, 2).contiguous() # (B, n, h, r, d//h)
        keys = self.conv_keys(feats).view(B, self.h, self.dim//self.h, N, -1).permute(0, 3, 1, 2, 4).contiguous() # (B, n, h, d//h, nkp)
        values = self.conv_values(feats).view(B, self.h, self.dim//self.h, N, -1).permute(0, 3, 1, 4, 2).contiguous() # (B, n, h, nkp, d//h) 

        logits = torch.matmul(querys, keys) * ((self.dim // self.h)**(-0.5)) # (B, n, h, r, nkp) 
        soft_logits = torch.softmax(logits, dim=-1)
        agg_feats = torch.matmul(soft_logits, values).permute(0, 1, 2, 4, 3).contiguous().view(B, N, -1, self.r) # (B, n, d, r) 
        agg_feats = agg_feats.permute(0,2,1,3).contiguous()# (B, n, d, r)
        agg_feats = self.attn_conv(agg_feats) + querys_iden# (B, d, N, r) 
        agg_feats = self.ffn_act(self.ffn(agg_feats) + agg_feats)
        return agg_feats


class Decoder_s(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.r = cfgs.up_rate
        self.disp_dim = cfgs.trans_dim
        self.dim = cfgs.out_dim
        self.rem = REM(cfgs, is_deform = True, need_repkpoints = True)
        self.kgm = KGM(cfgs)
        self.projector = nn.Conv2d(cfgs.trans_dim, cfgs.trans_dim, 1)
        self.attns = nn.ModuleList()
        for i in range(cfgs.trans_num):
            self.attns.append(MHA_o(cfgs))
        self.skip_mlp = nn.Sequential(
            nn.Conv1d(cfgs.out_dim + cfgs.trans_dim, cfgs.trans_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(cfgs.trans_dim, cfgs.trans_dim, 1),
            nn.ReLU(inplace=True)
        )
        self.disp_mlp = nn.Sequential(
            nn.Conv1d(cfgs.trans_dim, 64, 1),  
            nn.ReLU(inplace = True),
            nn.Conv1d(64, 3, 1)
        )
        
    def forward(self, pos, feat):
        B, _, N = pos.shape
        local_feat, RepKPoints, reg_loss = self.rem(pos, feat)
        KPQueries = self.kgm(pos, local_feat)
        qs = self.projector(KPQueries[1])
        ks = self.projector(RepKPoints[1])
        for m in self.attns:
            qs = m(qs, ks)
        disp_feat = qs.view(B, self.disp_dim, -1) # b, d, n * r
        disp_feat = self.skip_mlp(torch.cat([disp_feat, feat.unsqueeze(-1).repeat(1,1,1,self.r).view(B, self.dim, -1)], dim=1))
        disp = torch.tanh(self.disp_mlp(disp_feat))
        new_pos = pos.unsqueeze(-1).repeat(1,1,1,self.r).view(B, 3, -1) + disp # b, 3, n*r

        return new_pos, reg_loss
