import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import sys
import argparse
from models.repkpu import RepKPU, RepKPU_o
from cfgs.upsampling import parse_pu1k_args, parse_pugan_o_args, parse_pugan_args
from cfgs.utils import *
from dataset.dataset import PUDataset
import torch.optim as optim
from glob import glob
import open3d as o3d
from einops import repeat
from models.utils import *
import time
from datetime import datetime

def _normalize_point_cloud(pc):
    # b, n, 3
    centroid = torch.mean(pc, dim=1, keepdim = True) # b, 1, 3
    pc = pc - centroid # b, n, 3
    furthest_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0] # b, 1, 1
    pc = pc / furthest_distance
    return pc

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(_normalize_point_cloud(p1), _normalize_point_cloud(p2))
    d1 = torch.mean(d1)
    d2 = torch.mean(d2)
    return (d1 + d2)

def upsampling(args, model, input_pcd):
    pcd_pts_num = input_pcd.shape[-1]
    patch_pts_num = args.num_points
    sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
    seed = FPS(input_pcd, sample_num)
    patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
    patches, centroid, furthest_distance = normalize_point_cloud(patches)
    coarse_pts, _= model.forward(patches)
    coarse_pts = coarse_pts
    coarse_pts = centroid + coarse_pts * furthest_distance
    coarse_pts = rearrange(coarse_pts, 'b c n -> c (b n)').contiguous()
    coarse_pts = FPS(coarse_pts.unsqueeze(0), input_pcd.shape[-1]* args.up_rate)
    return coarse_pts

def _midpoint_interpolate(up_rate, sparse_pts):
    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * up_rate) + 1
    k = int(2 * up_rate)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    interpolated_pts = mid_pts
    interpolated_pts = FPS(interpolated_pts, up_pts_num)
    return interpolated_pts

# supprt 4x and 16x
def test(model, args):
    with torch.no_grad():
        model.eval()
        test_input_path = glob(os.path.join(args.input_dir, '*.xyz'))
        total_cd = 0
        counter = 0
        txt_result = []
        for i, path in enumerate(test_input_path):
            pcd = o3d.io.read_point_cloud(path)
            pcd_name = path.split('/')[-1]
            gt = torch.Tensor(np.asarray(o3d.io.read_point_cloud(os.path.join(args.gt_dir, pcd_name)).points)).unsqueeze(0).cuda()
            input_pcd = np.array(pcd.points)
            input_pcd = torch.from_numpy(input_pcd).float().cuda()
            input_pcd = rearrange(input_pcd, 'n c -> c n').contiguous()
            input_pcd = input_pcd.unsqueeze(0)

            input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)
            pcd_upsampled = upsampling(args, model, input_pcd)
            pcd_upsampled = centroid + pcd_upsampled * furthest_distance

            if args.r == 16:
                pcd_upsampled, centroid, furthest_distance = normalize_point_cloud(pcd_upsampled)
                pcd_upsampled = upsampling(args, model, pcd_upsampled)
                pcd_upsampled = centroid + pcd_upsampled * furthest_distance

            saved_pcd = rearrange(pcd_upsampled.squeeze(0), 'c n -> n c').contiguous()
            saved_pcd = saved_pcd.detach().cpu().numpy()
            save_folder = os.path.join(args.save_dir, 'xyz')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            np.savetxt(os.path.join(save_folder, pcd_name), saved_pcd, fmt='%.6f')
            
            
            cd = chamfer_sqrt(pcd_upsampled.permute(0,2,1).contiguous(), gt).cpu().item()  
            txt_result.append(f'{pcd_name}: {cd * 1e3}')    
            total_cd += cd
            counter += 1.0
        txt_result.append(f'overall: {total_cd/counter*1e3}')
        with open(os.path.join(args.save_dir,'cd.txt'), "w") as f:
            for ll in txt_result:
                f.writelines(ll+'\n')
    return total_cd/counter*1e3    

# Fiexed-to-Arbitrary
def test_flexible(model, args):
    with torch.no_grad():
        model.eval()
        test_input_path = glob(os.path.join(args.input_dir, '*.xyz'))
        total_cd = 0
        counter = 0
        txt_result = []
        for i, path in enumerate(test_input_path):
            pcd = o3d.io.read_point_cloud(path)
            pcd_name = path.split('/')[-1]
            gt = torch.Tensor(np.asarray(o3d.io.read_point_cloud(os.path.join(args.gt_dir, pcd_name)).points)).unsqueeze(0).cuda()
            input_pcd = np.array(pcd.points)
            input_pcd = torch.from_numpy(input_pcd).float().cuda()
            input_pcd = rearrange(input_pcd, 'n c -> c n').contiguous()
            target_num = int(args.r * input_pcd.shape[-1])
            input_pcd = input_pcd.unsqueeze(0)

            tmp_up_rate = float(args.r)
            if tmp_up_rate / 4.0 > 1.0:
                input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)
                input_pcd = _midpoint_interpolate(tmp_up_rate/4.0, input_pcd)
                input_pcd = centroid + input_pcd * furthest_distance

            input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)
            pcd_upsampled = upsampling(args, model, input_pcd)
            pcd_upsampled = centroid + pcd_upsampled * furthest_distance

            if pcd_upsampled.shape[-1] > target_num:
                pcd_upsampled = pcd_upsampled[:, :, (pcd_upsampled.shape[-1]-target_num):]

            print(pcd_upsampled.shape, gt.shape)
            
            saved_pcd = rearrange(pcd_upsampled.squeeze(0), 'c n -> n c').contiguous()
            saved_pcd = saved_pcd.detach().cpu().numpy()
            save_folder = os.path.join(args.save_dir, 'xyz')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            np.savetxt(os.path.join(save_folder, pcd_name), saved_pcd, fmt='%.6f')
            
            
            cd = chamfer_sqrt(pcd_upsampled.permute(0,2,1).contiguous(), gt).cpu().item()  
            txt_result.append(f'{pcd_name}: {cd * 1e3}')    
            total_cd += cd
            counter += 1.0
        txt_result.append(f'overall: {total_cd/counter*1e3}')
        with open(os.path.join(args.save_dir,'cd.txt'), "w") as f:
            for ll in txt_result:
                f.writelines(ll+'\n')

    return total_cd/counter*1e3    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing Arguments')
    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    parser.add_argument('--r', default=4, type=int, help='upsampling rate')
    parser.add_argument('--o', action='store_true', help='using original model')
    parser.add_argument('--flexible', action='store_true', help='aribitrary scale?')
    parser.add_argument('--input_dir', default='./output', type=str, help='path to folder of input point clouds')
    parser.add_argument('--gt_dir', default='./output', type=str, help='path to folder of gt point clouds')
    parser.add_argument('--save_dir', default='pcd', type=str, help='save upsampled point cloud and results')
    parser.add_argument('--ckpt', default='./output', type=str, help='checkpoints')
    args = parser.parse_args()
    
    if args.dataset == 'pugan':
        if args.o:
            reset_model_args(parse_pugan_o_args(), args) 
            model = RepKPU_o(args)
        else:
            reset_model_args(parse_pugan_args(), args)
            model = RepKPU(args)
    else:
        reset_model_args(parse_pu1k_args(), args)
        model = RepKPU(args)
    
    model = model.cuda()
    model.load_state_dict(torch.load(args.ckpt))
    if not args.flexible:
        test(model, args)
    else:
        test_flexible(model, args)
    