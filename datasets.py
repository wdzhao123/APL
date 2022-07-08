import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

class ImageDataset_test(Dataset):
    def __init__(self, root):
        self.input_path = root

        self.transform = transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor(),
        ])
        self.files = sorted(glob.glob(root + '/*.*'))

    def __getitem__(self, index):
        image_name = os.path.join(self.input_path, str(index+1) + '.bmp')   # CUHK and DUT
        # image_name = os.path.join(self.input_path, str(index + 1) + '.png')    #DP
        image = Image.open(image_name)
        transformed_img = self.transform(image)
        sample = {'image': transformed_img}
        return sample

    def __len__(self):
        return len(self.files)
def Get_dataloader_test(path,batch):
    test_dataloader = DataLoader(
        ImageDataset_test(path),
        batch_size=batch, shuffle=False, num_workers=2, drop_last=True)
    return test_dataloader

class ImageDataset_train(Dataset):
    def __init__(self, root):
        self.input_path = root + '/1204source'
        self.mask_path = root + '/1204gt'
        self.clear_path = root + '/dis_images/clear'

        self.transform = transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((160,160)),
            transforms.ToTensor(),
        ])
        self.files = sorted(glob.glob(self.input_path + '/*.*'))


    def __getitem__(self, index):
        image_name = os.path.join(self.input_path, str(index+1) + '.bmp')
        image = Image.open(image_name).convert('RGB')
        mask_name = os.path.join(self.mask_path, str(index+1) + '.bmp')
        mask = Image.open(mask_name)
        clear_name = os.path.join(self.clear_path, str(random.randint(1,500)) + '.bmp')
        clear = Image.open(clear_name).convert('RGB')

        transformed_img = self.transform(image)
        transformed_mask = self.transform_mask(mask)
        transformed_clear = self.transform(clear)

        sample = {'image': transformed_img, 'mask': transformed_mask, 'clear': transformed_clear}
        return sample

    def __len__(self):
        return len(self.files)
def Get_dataloader_train(path,batch):
    train_dataloader = DataLoader(ImageDataset_train(path),batch_size=batch, shuffle=True, num_workers=2, drop_last=True)
    return train_dataloader