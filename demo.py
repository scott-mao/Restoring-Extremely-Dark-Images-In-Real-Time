import numpy as np
import os
import shutil
import torch
from torch.utils.data import DataLoader
import glob
from common_classes import load_data, run_test
from network import Net

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

save_images = 'demo_restored_images'

shutil.rmtree(save_images, ignore_errors = True)
os.makedirs(save_images)

test_files = glob.glob('demo_imgs/*.ARW') 
dataloader_test = DataLoader(load_data(test_files), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model = Net()
print('\n Network parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model = model.to(device)
print('Device on GPU: {}'.format(next(model.parameters()).is_cuda))
checkpoint = torch.load('demo_imgs/weights', map_location=device)
model.load_state_dict(checkpoint['model'])

run_test(model, dataloader_test, save_images)
print('Restored images saved in DEMO_RESTORED_IMAGES directory')
