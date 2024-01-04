import torch
from scipy.io import savemat
from PIL import Image
import torchvision.transforms.functional as tf

from models import GeneratorUNet
import time
import sys
import os

##############
# Model loaded
##############
net = GeneratorUNet(1, 1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
net.to(device=device)


if __name__ == '__main__':
    # matlab input
    # img = sys.argv[1]
    # path = sys.argv[2]
    
    path = os.getcwd()
    img = os.getcwd()+'\\test.jpg'
    net.load_state_dict(torch.load(path+'\\generator_190.pth.tar', map_location=device))
    t_start = time.time()
    img = Image.open(img).convert('L')
    img = tf.to_tensor(img)
    img = img.to(device=device, dtype=torch.float32).reshape(1, 1, 512, 512)
    with torch.no_grad():
        mask_pred = net(img)
    mask_pred = (mask_pred.cpu().numpy().squeeze())
    mask_input = img.cpu().numpy().squeeze()
    t_end = time.time()
    print('run time is：', t_end-t_start)
    
    # save as mat
    savemat('test.mat', {'test': mask_pred})

    # max-min normalization
    mask_pred = (mask_pred - mask_pred.min()) / (mask_pred.max() - mask_pred.min())
    mask_pred = mask_pred * 255   
    mask_pred = mask_pred.astype('uint8')
    mask_pred = Image.fromarray(mask_pred)
    mask_pred.show()
    mask_pred.save('output_test.jpg')
    




