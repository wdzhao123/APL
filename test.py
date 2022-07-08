import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from datasets import Get_dataloader_test
from models.DBDNet import CNN_for_DBD
from models.DeblurNet import CNN_for_Generator
import torchvision.transforms as transforms


def test_deblur(stict1, stict2, result_save_path,image_path):
    print("Model Deblur Start Testing ... ...")
    DBD = CNN_for_DBD().cuda()
    DBD.load_state_dict(torch.load(stict1))
    DBD.eval()
    generator = CNN_for_Generator().cuda()
    generator.load_state_dict(torch.load(stict2))
    generator.eval()

    dataloader = Get_dataloader_test(image_path, batch = 1)
    for i,sample in enumerate(dataloader):
        image= sample['image']
        image=Variable(image).cuda()
        dbd_result = DBD(image)
        synimage = generator(image, dbd_result)
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        m = 0
        synimage[m, 0, :, :] = synimage[m, 0, :, :] * std[0] + mean[0]
        synimage[m, 1, :, :] = synimage[m, 1, :, :] * std[1] + mean[1]
        synimage[m, 2, :, :] = synimage[m, 2, :, :] * std[2] + mean[2]
        ones = torch.ones_like(synimage)
        zeros = torch.zeros_like(synimage)
        image_size = synimage.size()
        for p in range(image_size[1]):
            for q in range(image_size[2]):
                for n in range(image_size[3]):
                    if synimage[m, p, q, n] > ones[m, p, q, n]:
                        synimage[m, p, q, n] = ones[m, p, q, n]
                    elif synimage[m, p, q, n] < zeros[m, p, q, n]:
                        synimage[m, p, q, n] = zeros[m, p, q, n]
        synimage = synimage.squeeze()
        synimage = synimage.detach().cpu().float().numpy()
        synimage = np.uint8(synimage * 255)
        synimage = np.transpose(synimage, (1, 2, 0))
        synimage = Image.fromarray(synimage)
        synimage.save(os.path.join(result_save_path, str(i + 1) + '_deb.bmp'))
    print("End of Deblur Testing")

def test_DBD(stict1, mask_save_path,image_path):
    print("Model DBD Start Testing ... ...")
    DBD = CNN_for_DBD().cuda()
    DBD.load_state_dict(torch.load(stict1))
    DBD.eval()
    dataloader = Get_dataloader_test(image_path, 1)
    for i,sample in enumerate(dataloader):
        image = sample['image']
        image=Variable(image).cuda()
        dbd_result = DBD(image)
        os.makedirs(mask_save_path, exist_ok=True)
        dbd_result = dbd_result.cpu()
        dbd_result = dbd_result[0, :, :, :]
        dbd_result = torch.squeeze(dbd_result)
        img = transforms.ToPILImage()(dbd_result)
        img.save(os.path.join(mask_save_path, str(i + 1) + '.bmp'))
    print("End of DBD Testing")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--stict1', default='./saved_models/DBD.pth',type=str)
    parser.add_argument('--stict2', default='./saved_models/deblur.pth',type=str)
    parser.add_argument('--result_save_path', default='./result/test_CUHK', type=str)
    parser.add_argument('--image_path', default='./dataset/test/CUHK-source', type=str)
    args=parser.parse_args()

    ##### test #####
    test_deblur(args.stict1, args.stict2, args.result_save_path, args.image_path)
    test_DBD(args.stict1,args.result_save_path, args.image_path)
