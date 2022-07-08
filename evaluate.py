import argparse
import os
import numpy as np
from PIL import Image
from config import*

def eval(image_save_path,test_gt_path):
    files=os.listdir(test_gt_path)
    maes=0
    mses=0
    psnrs=0
    ssims=0
    for file in files:
        image1=image_save_path+'/'+file
        gt1=test_gt_path+'/'+file
        image1 = Image.open(image1).convert('RGB')
        image1 = image1.resize((160, 160))
        image1 = np.array(image1)
        image1 = image1.astype(float)/255.0
        gt1 = Image.open(gt1).convert('RGB')
        gt1 = gt1.resize((160, 160))
        gt1 = np.array(gt1)
        gt1 = gt1.astype(float)/255.0

        psnr, ssim = MSE_PSNR_SSIM((gt1).astype(np.float64), (image1).astype(np.float64))
        mae = MAE((gt1).astype(np.float64), (image1).astype(np.float64))
        maes += mae
        psnrs += psnr
        ssims += ssim
    mae1=maes/len(files)
    psnr1=psnrs/len(files)
    ssim1=ssims/len(files)
    return psnr1, ssim1, mae1
def MAE(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])
def MSE_PSNR_SSIM(img1, img2):
    mse_ = np.mean( (img1 - img2) ** 2 )
    if mse_ == 0:
        return 100
    PIXEL_MAX = 1
    return 10 * math.log10(PIXEL_MAX / mse_), measure.compare_ssim(img1,
                                                                         img2, data_range=PIXEL_MAX,
                                                                         multichannel=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--image_save_path', default='./log/test-DP', type=str)
    parser.add_argument('--test_gt_path', default='./dataset/test/DP-target', type=str)
    args=parser.parse_args()

    psnr, ssim, mae = eval(args.image_save_path, args.test_gt_path)
    print('psnr:%.3f,ssim:%.3f, mae:%.3f' % (psnr, ssim, mae))
