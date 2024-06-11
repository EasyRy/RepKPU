import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import sys
import argparse
from models.repkpu import RepKPU
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

def train(model, args):
    set_seed(args.seed)
    start = time.time()

    # dataloader
    train_dataset = PUDataset(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   shuffle=True,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers)

    # set up folders for checkpoints and logs
    str_time = datetime.now().isoformat()
    output_dir = os.path.join(args.out_path, str_time)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = get_logger('train', log_dir)
    logger.info('Experiment ID: %s' % (str_time))

    # model
    logger.info('========== Build Model ==========')
    model = model.cuda()
    model.train()

    # optimizer
    deform_params = [v for k, v in model.named_parameters() if 'deform' in k]
    other_params = [v for k, v in model.named_parameters() if 'deform' not in k]
    assert args.optim in ['adam', 'sgd']
    if args.optim == 'adam':
        optimizer = optim.Adam([{'params': other_params},
                                {'params': deform_params, 'lr': args.lr*0.1}], 
                                lr=args.lr, weight_decay=args.weight_decay)
    else:
        args.lr = args.lr * 100
        optimizer = optim.SGD([{'params': other_params},
                                {'params': deform_params, 'lr': args.lr * 0.1}], lr=args.lr)
    
    # scheduler
    scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.05**(1/150))
    #scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    
    logger.info(args)
    logger.info('========== Begin Training ==========')
    best_cd = 10000
    
    
    for epoch in range(args.epochs):
        model.train()
        logger.info('********* Epoch %d *********' % (epoch + 1))
       
        epoch_loss = 0.0
        for i, (input_pts, gt_pts, radius) in enumerate(train_loader):
            # (b, n, 3) -> (b, 3, n)
            input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
            gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()
            gen_pts, reg_loss = model.forward(input_pts)
            loss = get_cd_loss(args, gen_pts, gt_pts)
            if reg_loss != None:
                loss_all = loss + reg_loss
            else:
                loss_all = loss
            epoch_loss += loss.item()
            # update parameters
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            

            # log
            if (i+1) % args.print_rate == 0:
                logger.info("epoch: %d/%d, iters: %d/%d, loss: %f" %
                      (epoch + 1, args.epochs, i + 1, len(train_loader), epoch_loss / (i+1)))

        # lr scheduler
        scheduler_steplr.step()

        # log
        interval = time.time() - start
        
        cd_now = val(model, args)
        if cd_now < best_cd:
            best_cd = cd_now
            model_name = 'ckpt-best.pth'
            model_path = os.path.join(ckpt_dir, model_name)
            torch.save(model.state_dict(), model_path)

        logger.info("epoch: %d/%d, avg epoch loss: %f, cd_last: %f, cd_best: %f, time: %d mins %.1f secs" %
          (epoch + 1, args.epochs, epoch_loss / len(train_loader), cd_now, best_cd, interval / 60, interval % 60))
        

def val(model, args):
    with torch.no_grad():
        model.eval()
        test_input_path = glob(os.path.join(args.input_dir, '*.xyz'))
        total_cd = 0
        counter = 0
        
        for i, path in enumerate(test_input_path):
                # each time upsample one point cloud
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

            cd = chamfer_sqrt(pcd_upsampled.permute(0,2,1).contiguous(), gt).cpu().item()  
                
            total_cd += cd
            counter += 1.0
       
    return total_cd/counter*1e3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    args = parser.parse_args()
    if args.dataset == 'pu1k':
        reset_model_args(parse_pu1k_args(), args)
    else:
        reset_model_args(parse_pugan_args(), args)
    
    # 
    model = RepKPU(args)
    train(model, args)