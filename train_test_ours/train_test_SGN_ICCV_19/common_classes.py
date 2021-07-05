import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import models
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import normalized_root_mse as NRMSE
import rawpy
import glob
import imageio

def define_weights(num):
    weights = np.float32((np.logspace(0,num,127, endpoint=True, base=10.0)))
    weights = weights/np.max(weights)
    weights = np.flipud(weights).copy()    
    return weights

def get_na(bins,weights,img_loww,amp=5):
    H,W = img_loww.shape
    arr = img_loww*1
    selection_dict = {weights[0]: (bins[0]<=arr)&(arr<bins[1])}
    for ii in range(1,len(weights)):
        selection_dict[weights[ii]] = (bins[ii]<=arr)&(arr<bins[ii+1])
    mask = np.select(condlist=selection_dict.values(), choicelist=selection_dict.keys())
   
    mask_sum1 = np.sum(mask,dtype=np.float64)
    
    na1 = np.float32(np.float64(mask_sum1*0.01*amp)/np.sum(img_loww*mask,dtype=np.float64))
# As in SID max amplification is limited to 300
    if na1>300.0:
        na1 = np.float32(300.0)
    if na1<1.0:
        na1 = np.float32(1.0)
    
    selection_dict.clear()

    return na1


def part_init(gt_files,train_files,num_print,filename,gt_amp=False):

    file_line = open(filename, 'w')
    bins = np.float32((np.logspace(0,8,128, endpoint=True, base=2.0)-1))/255.0
    print('\nEdges:{}, dtype:{}\n'.format(bins,bins.dtype), file = file_line)
    weights5 = define_weights(5)
    print('------- weights: {}\n'.format(weights5), file = file_line)

    gt_list = []
    train_list = []
    mean = 0
    
    for i in range(len(gt_files)):
        raw = rawpy.imread(gt_files[i])
        img_gt = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16).copy()
        raw.close()            
        img_gtt=np.float32(img_gt/65535.0)
        h,w,_ = img_gtt.shape
        
        correct_dim_flag = False
        if h%32!=0:
            print('Correcting the 1st dimension.')
            h = (h//32)*32
            img_gtt = img_gtt[:h,:,:]
            correct_dim_flag = True
        
        if w%32!=0:
            print('Correcting the 2nd dimension.')
            w = (w//32)*32
            img_gtt = img_gtt[:,:w,:]
            correct_dim_flag = True
            
        gt_list.append(img_gtt)
        
        raw = rawpy.imread(train_files[i])
        img = raw.raw_image_visible.astype(np.float32).copy()
        raw.close()
        
        if correct_dim_flag:
            img = img[:h,:w]        
        
        img_loww = (np.maximum(img - 512,0)/ (16383 - 512))       
        
        na5 = get_na(bins,weights5,img_loww)
        
        if gt_files[i][-7]=='3':
            ta=300
        else:
            ta=100
        
        H,W = img_loww.shape    
        a = np.float32(np.float64(H*W*0.01)/np.sum(img_loww,dtype=np.float64))
        
        if gt_amp:
            img_loww = (img_loww*ta)
            print('...using gt_amp : {}'.format(gt_files[i][-17:]), file = file_line)
        else:
            img_loww = (img_loww*na5)
            print('...using na5 : {}'.format(gt_files[i][-17:]), file = file_line)
            
        train_list.append(img_loww)
        mean += np.mean(img_loww[0::2,1::2],dtype=np.float32)

        if (i+1)%num_print==0:
            print('... files loading : {}'.format(gt_files[i][-17:]))
            print('Image {} base_amp: {}, gt_amp: {}, Our_Amp:{}'.format(i+1,a,ta,na5))
        
        print('Image {} base_amp: {}, gt_amp: {}, Our_Amp:{}'.format(i+1,a,ta,na5), file = file_line)
   
    print('Files loaded : {}/{}, channel mean: {}'.format(len(train_list), len(gt_files), mean/len(train_list)))
    file_line.close()
    return gt_list, train_list
    
    
################ DATASET CLASS
class load_data(Dataset):
    """Loads the Data."""
    
    def __init__(self, train_files, gt_files, filename, num_print, gt_amp=True, training=True):        
        
        self.training = training
        if self.training:
            print('\n...... Train files loading\n')
            self.gt_list, self.train_list = part_init(gt_files,train_files,num_print,filename,gt_amp)        
            print('\nTrain files loaded ......\n')
        else:
            print('\n...... Test files loading\n')
            self.gt_list, self.train_list = part_init(gt_files,train_files,num_print,filename,gt_amp)        
            print('\nTest files loaded ......\n')
        
    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
    
        img_gtt = self.gt_list[idx]
        img_loww = self.train_list[idx]
        
        H,W = img_loww.shape
        
        if self.training:
            i = random.randint(0, (H-512-2)//2)*2
            j = random.randint(0,(W-512-2)//2)*2

            img_low = img_loww[i:i+512,j:j+512]
            img_gt = img_gtt[i:i+512,j:j+512,:]
            
            if random.randint(0, 100)>50:
                img_gt = np.fliplr(img_gt).copy()
                img_low = np.fliplr(img_low).copy()

            if random.randint(0, 100)<20:
                img_gt = np.flipud(img_gt).copy()
                img_low = np.flipud(img_low).copy()
        else:
            img_low = img_loww
            img_gt = img_gtt
            
        gt = torch.from_numpy((np.transpose(img_gt, [2, 0, 1]))).float()
        low = torch.from_numpy(img_low).float().unsqueeze(0)            
        
        return low, gt
        

        
#################### perceptual loss

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        
        self.l1 = torch.nn.L1Loss()
        
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x].eval())
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x].eval())
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x].eval())
#        for x in range(16, 23):
#            self.slice4.add_module(str(x), vgg_pretrained_features[x].eval())
        for name, param in self.named_parameters():
            param.requires_grad = False
            print(name,' grad of VGG set to false !!')
    
    def VGGfeatures(self, x):
        
        x = self.slice1(x)
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
#            h = self.slice4(h)
#            h_relu4_3 = h
            
        return relu2_2, relu3_3


    def forward(self, ip, target, which='relu2'):
        
        ip = (ip-self.mean) / self.std
        target = (target-self.mean) / self.std
    
        ip_relu2_2, ip_relu3_3 = self.VGGfeatures(ip)
        target_relu2_2, target_relu3_3 = self.VGGfeatures(target)
    
        if which=='relu2':
            loss = self.l1(ip_relu2_2,target_relu2_2)
        elif which=='relu3':
            loss = self.l1(ip_relu3_3,target_relu3_3)
        elif which=='both':
            loss = self.l1(ip_relu2_2,target_relu2_2) + self.l1(ip_relu3_3,target_relu3_3)
        else:
            raise NotImplementedError('Incorrect WHICH in perceptual loss.')
        
        return loss
        

    
######### testing


def run_test(model, dataloader_test, iteration, save_images, save_csv_files, metric_average_filename, mode, training=True):
    psnr = ['PSNR']
    ssim = ['SSIM']
    C_pred = ['Colorfulness_pred']
    C_gt = ['Colorfulness_gt']
    
    with torch.no_grad():
        model.eval()
        for image_num, img in enumerate(dataloader_test):
            low = img[0].to(next(model.parameters()).device)
            gt = img[1]
            pred = model(low)
            
            pred = (np.clip(pred[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
            gt = (np.clip(gt[0].detach().cpu().numpy().transpose(1,2,0),0,1)*255).astype(np.uint8)
            psnr_img = PSNR(pred,gt)
            ssim_img = SSIM(pred,gt,multichannel=True)
            c_gt = 0
            c_pred = 0
            
            cond = True
            if training:
                cond = image_num in [0,1,2,3,7,10,11,12,13,19,20,30,35,41,46,47,48] # During training testing will be done only for few test images. Include those in this list.
            
            if cond:
                imageio.imwrite(os.path.join(save_images,'{}_{}_gt_C_{}.jpg'.format(image_num,iteration,c_gt)), gt)
                imageio.imwrite(os.path.join(save_images,'{}_{}_psnr_{}_ssim_{}_C_{}.jpg'.format(image_num,iteration, psnr_img, ssim_img,c_pred)), pred)
            
            psnr.append(psnr_img)
            ssim.append(ssim_img)
            C_pred.append(c_pred)
            C_gt.append(c_gt)
            
    np.savetxt(os.path.join(save_csv_files,'Metrics_iter_{}.csv'.format(iteration)), [p for p in zip(psnr,ssim,C_pred,C_gt)], delimiter=',', fmt='%s')
    
    psnr_avg = sum(psnr[1:]) / len(psnr[1:])
    ssim_avg = sum(ssim[1:]) / len(ssim[1:])
    c_gt_avg = sum(C_gt[1:]) / len(C_gt[1:])
    c_pred_avg = sum(C_pred[1:]) / len(C_pred[1:])

    f =  open(metric_average_filename, mode)
    f.write('-- psnr_avg:{}, ssim_avg:{}, c_gt_avg:{}, c_pred_avg:{}, iter:{}\n'.format(psnr_avg,ssim_avg,c_gt_avg,c_pred_avg,iteration))
    print('metric average printed.')        
    f.close()

    return
    
    

        
