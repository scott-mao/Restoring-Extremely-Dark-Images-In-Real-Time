opt={'base_lr':1e-4}
opt['reduce_lr_by'] = 0.1
opt['atWhichReduce'] = [300000]
opt['batch_size'] = #Enter batch size
opt['atWhichSave'] = [2,100002,150002,200002,250002,300002,350002,400002,450002,500002, 600000,700000,800000,900000,1000000]
opt['iterations'] = 1000005
dry_run_iterations = 100
dry_run = False
metric_average_file = 'metric_average.txt'
test_amplification_file = 'test_amplification.txt'
train_amplification_file = 'train_amplification.txt'
# These are folders
save_weights = 'weights'
save_images = 'images' # make this change below also
save_csv_files = 'csv_files'

import numpy as np
import os
import shutil
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import glob

from common_classes import load_data, run_test
from network import Net
import argparse

# Argument for EDSR
parser = argparse.ArgumentParser(description='EDSR')
parser.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--scale', type=str, default=2,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=256,
                    help='output patch size')
parser.add_argument('--n_colors', type=int, default=4,
                    help='number of input color channels to use')
parser.add_argument('--o_colors', type=int, default=3,
                    help='number of output color channels to use')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

shutil.rmtree(metric_average_file, ignore_errors = True)
shutil.rmtree(test_amplification_file, ignore_errors = True)
shutil.rmtree(train_amplification_file, ignore_errors = True)

shutil.rmtree(save_weights, ignore_errors = True)
shutil.rmtree(save_images, ignore_errors = True)
shutil.rmtree(save_csv_files, ignore_errors = True)

os.makedirs(save_weights)
os.makedirs(save_images)
os.makedirs(save_csv_files)

train_files = glob.glob('/SID_cvpr_18_dataset/Sony/short/0*_00_0.1s.ARW')
train_files +=glob.glob('/SID_cvpr_18_dataset/Sony/short/2*_00_0.1s.ARW')
if dry_run:
    train_files = train_files[:2]
    opt['iterations'] = dry_run_iterations
    
gt_files = []
for x in train_files:
    gt_files += glob.glob('/SID_cvpr_18_dataset/Sony/long/*'+x[-17:-12]+'*.ARW')
    
dataloader_train = DataLoader(load_data(train_files,gt_files,train_amplification_file,20,gt_amp=True,training=True), batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

test_files = glob.glob('/SID_cvpr_18_dataset/Sony/short/1*_00_0.1s.ARW') 
if dry_run:
    test_files = test_files[:2]
    
gt_files = []
for x in test_files:
    gt_files = gt_files+ glob.glob('/SID_cvpr_18_dataset/Sony/long/*'+x[-17:-12]+'*.ARW')
dataloader_test = DataLoader(load_data(test_files,gt_files,test_amplification_file,2,gt_amp=True,training=False), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

for i,img in enumerate(dataloader_train):    
    print('Input image size : {}, GT image size : {}'.format(img[0].size(), img[1].size()))    
    break
    
############ Training Begins

device = torch.device("cuda")
model = Net(args)
print(model)
print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))

iter_num = 0
l1_loss = torch.nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=opt['base_lr'])
optimizer.zero_grad()
loss_list = ['L1_loss']
loss_iter_list = ['Iteration']
iter_LR = ['Iter_LR']
            

while iter_num<opt['iterations']:
    for _, img in enumerate(dataloader_train):
        low = img[0].to(device)
        gt = img[1].to(device)
        model.train()
        pred = model(low)
        iter_num +=1
        #loss2 = feature_loss(pred, gt, which='relu2')
        loss1 = l1_loss(pred,gt)
        #print('l1,vgg loss: ',loss1.item(), loss2.item())
        loss = loss1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()        
        
        if iter_num>opt['iterations']:
            break        
        
        if iter_num%10==0:
            print(iter_num)
            if iter_num%100==0:
                loss_list.append('{},{},{}'.format(loss1.item(),-1,-1))
                loss_iter_list.append(iter_num)
                iter_LR.append(optimizer.param_groups[0]['lr'])
                
        if iter_num in opt['atWhichReduce']:
            for group in optimizer.param_groups:
                old_lr = group['lr']
                group['lr'] = opt['reduce_lr_by']*old_lr
                if group['lr']<1e-5:
                    group['lr']=1e-5
                print('Changing LR from {} to {}'.format(old_lr,group['lr']))
                
        if iter_num in opt['atWhichSave']:
            print('testing......')
            if iter_num == opt['atWhichSave'][0]:
                mode = 'w'
            else:
                mode = 'a'
            run_test(model, dataloader_test, iter_num, save_images, save_csv_files, metric_average_file, mode, training=True)
            torch.save({'model': model.state_dict()},os.path.join(save_weights,'weights_{}'.format(iter_num)))


np.savetxt(os.path.join(save_csv_files,'loss_curve.csv'),[p for p in zip(loss_iter_list,loss_list,iter_LR)],delimiter=',',fmt='%s')

   
                
        
