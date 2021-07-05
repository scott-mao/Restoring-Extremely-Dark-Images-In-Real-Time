opt={'base_lr':1e-4}
opt['reduce_lr_by'] = 0.1
opt['atWhichReduce'] = [500000]
opt['batch_size'] = #Enter batch size
opt['atWhichSave'] = [2,100002,150002,200002,250002,300002,350002,400002,450002,500002,550000, 600000,650002,700002,750000,800000,850002,900002,950000,1000000]
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
import argparse

from common_classes import load_data, run_test
import network



parser = argparse.ArgumentParser()
# Initialization parameters
parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
parser.add_argument('--in_channels', type = int, default = 1, help = 'input channels for generator')
parser.add_argument('--out_channels', type = int, default = 3, help = 'output channels for generator')
parser.add_argument('--start_channels', type = int, default = 32, help = 'start channels for generator')
parser.add_argument('--m_block', type = int, default = 2, help = 'the additional blocks used in mainstream')
parser.add_argument('--init_type', type = str, default = 'normal', help = 'initialization type of generator')
parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
optt = parser.parse_args()

model = network.SGN(optt)
network.weights_init(model, init_type = optt.init_type, init_gain = optt.init_gain)
print('Generator is created!')


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
    
dataloader_train = DataLoader(load_data(train_files,gt_files,train_amplification_file,20,gt_amp=True,training=True), batch_size=opt['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

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
print(model)
print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))

iter_num = 0
l1_loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt['base_lr'])
optimizer.zero_grad()
loss_list = ['L1_loss,Feature_loss,MS_SSIM']
loss_iter_list = ['Iteration']
iter_LR = ['Iter_LR']
            

while iter_num<opt['iterations']:
    for _, img in enumerate(dataloader_train):
        low = img[0].to(device)
        gt = img[1].to(device)
        model.train()
        pred = model(low)
        iter_num +=1
        loss1 = l1_loss(pred,gt)
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
                
        if iter_num in opt['atWhichSave']:
            print('testing......')
            if iter_num == opt['atWhichSave'][0]:
                mode = 'w'
            else:
                mode = 'a'
            run_test(model, dataloader_test, iter_num, save_images, save_csv_files, metric_average_file, mode, training=True)
            torch.save({'model': model.state_dict()},os.path.join(save_weights,'weights_{}'.format(iter_num)))
            
        if iter_num in opt['atWhichReduce']:
            for group in optimizer.param_groups:
                old_lr = group['lr']
                group['lr'] = opt['reduce_lr_by']*old_lr
                if group['lr']<1e-5:
                    group['lr']=1e-5
                print('Changing LR from {} to {}'.format(old_lr,group['lr']))

np.savetxt(os.path.join(save_csv_files,'loss_curve.csv'),[p for p in zip(loss_iter_list,loss_list,iter_LR)],delimiter=',',fmt='%s')


        
