import os
import torch
from torch.utils.data import Dataset
#import glob
import numpy as np
import random as ra
from utils.util import read_filepaths
from PIL import Image, ImageOps
import cv2
from matplotlib import cm
from torchvision import transforms
from batchgenerators.transforms import spatial_transforms
from skimage.util import random_noise

COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}
DatasetDIC = {'ACT': 0, 'BIMCV': 1, 'CHEXPERT': 2, 'CHS':3, 'CRXNIH': 4, 'HM1': 5, 'HM2': 6, 'MIMIC': 7, 'MOS': 8, 'NIH': 9}
def do_augmentation(image_tensor):
    array, _ = spatial_transforms.augment_mirroring(image_tensor, axes=(1, 2))
    augmented = array[None, ...]
    r_range = (0, (3 / 360.) * 2 * np.pi)
    cval = 0.
        
    augmented, _ = spatial_transforms.augment_spatial(
        augmented, seg=np.ones_like(augmented), patch_size=[augmented.shape[2],augmented.shape[3]],
        do_elastic_deform=True, alpha=(0., 100.), sigma=(8., 13.),
        do_rotation=True, angle_x=r_range, angle_y=r_range, angle_z=r_range,
        do_scale=True, scale=(.9, 1.1),
        border_mode_data='constant', border_cval_data=cval,
        order_data=3,
        p_el_per_sample=0.5,
        p_scale_per_sample=.5,
        p_rot_per_sample=.5,
        random_crop=False
    )
    return augmented

class COVIDxDataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, mode, n_classes=3, dataset_path='./datasets', dim=(224, 224), pre_processing = 'None'):
        self.root = str(dataset_path) + '/' + mode + '/'

        self.CLASSES = n_classes
        self.dim = dim
        self.COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}
        self.pre_processing = pre_processing
        testfile = '/home/huaxu@gaps_domain.ssr.upm.es/projects/COVID_BayesianNET/data/test.txt'
        trainfile = '/home/huaxu@gaps_domain.ssr.upm.es/projects/COVID_BayesianNET/data/train.txt'
        if (mode == 'train'):
            self.paths, self.labels, _ = read_filepaths(trainfile)
        elif (mode == 'test'):
            self.paths, self.labels, _ = read_filepaths(testfile)
        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_tensor = self.load_image(self.root + self.paths[index])
        label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)
        image_tensor = image_tensor.numpy()

        if ra.random()>0.5: #add noise to the image 随机加入高斯噪声
            image_tensor = random_noise(image_tensor, mode='gaussian', mean=0.015, var = 0.015)
            
        if ((label_tensor.numpy() == 2 and ra.random()>0.17) or (label_tensor.numpy() ==0 and ra.random()>0.5)) and self.mode == 'train':#apply data augmentation only for COVID:if label_tensor.numpy() == 2  and self.mode == 'train'
            augmented_tensor = do_augmentation(image_tensor) 
            augmented_tensor = torch.from_numpy(augmented_tensor) 
            augmented_tensor = torch.squeeze(augmented_tensor, dim=0)
            final_tensor = augmented_tensor
        else:
            final_tensor = torch.FloatTensor(image_tensor)
        return final_tensor, label_tensor

    def load_image(self, img_path):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
    
        if self.pre_processing == 'None':
            image = cv2.imread(img_path)
            img_adapteq = Image.fromarray(image.astype('uint8'), 'RGB')
        elif self.pre_processing == 'Equalization': #直方图均衡化
            image = cv2.imread(img_path)
            image2 = np.copy(image)
            image2[image2>0]=255 # 将图像中大于0的像素值设为255，即将图像转换为二值图像
            image2 = image2[:,:,0] # 将图像转换为灰度图，[:,:,0]表示取第一个通道,图像的shape为(512,512,3),转换后的shape为(512,512)，即灰度图
            mask = Image.fromarray(image2.astype('uint8')) #将numpy数组转换为PIL图像
            img_adapteq = Image.fromarray(image.astype('uint8'), 'RGB') #将numpy数组转换为PIL图像
            img_adapteq = ImageOps.equalize(img_adapteq,mask=mask) #对图像进行直方图均衡化，生成的图像的亮度将会根据mask的亮度进行调整
        elif self.pre_processing == 'CLAHE':#对比度受限的自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=4,tileGridSize=(8,8)) #创建CLAHE对象，clipLimit为对比度的限制，tileGridSize为每个块的大小
            image = cv2.imread(img_path, cv2.IMREAD_COLOR) #读取图像
            image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab) #将图像转换为Lab颜色空间
            image[:,:,0] = clahe.apply(image[:,:,0]) #对亮度通道进行CLAHE处理
            image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB) #将图像转换为RGB颜色空间
            img_adapteq = Image.fromarray(image.astype('uint8'), 'RGB') #将numpy数组转换为PIL图像
            #equalization 和 CLAHE 都是图像增强的方法 主要为了增加图像的对比度，使图像更加清晰，从而提高图像的识别率
            #而transform是数据增强的方法，主要是为了增加数据的多样性，提高模型的泛化能力

        preprocess = transforms.Compose([
            transforms.Resize(self.dim[0]),
            transforms.CenterCrop(self.dim[0]),
            transforms.ToTensor(),#normalize to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        image_tensor = preprocess(img_adapteq)
        
        return image_tensor
    
class COVIDxDataset_DA(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, mode, n_classes=3, dataset_path='./datasets', dim=(224, 224), pre_processing = 'None'):
        self.root = str(dataset_path) #+ '/' + mode + '/'

        self.CLASSES = n_classes
        self.dim = dim
        self.COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}
        self.pre_processing = pre_processing
        testfile = '/home/huaxu@gaps_domain.ssr.upm.es/projects/COVID_BayesianNET/data/test.txt'
        trainfile = '/home/huaxu@gaps_domain.ssr.upm.es/projects/COVID_BayesianNET/data/train.txt'
        if (mode == 'train'):
            self.paths, self.labels, self.dbs = read_filepaths(trainfile)
        elif (mode == 'test'):
            self.paths, self.labels, self.dbs = read_filepaths(testfile)
        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_tensor = self.load_image(self.root + self.paths[index])
        label_tensor = self.COVIDxDICT[self.labels[index]]
        image_tensor = image_tensor.numpy()
        label_db = DatasetDIC[self.paths[index].split("_")[0]] #get the dataset index，即数据集来源的索引
        labels = torch.tensor([label_tensor,label_db],dtype=torch.long)
        if ra.random()>0.5:
            image_tensor = random_noise(image_tensor, mode='gaussian', mean=0.015, var = 0.015)
            
        if ((label_tensor == 2 and ra.random()>0.17) or (label_tensor ==0 and ra.random()>0.5)) and self.mode == 'train':
            #训练集中只对COVID-19和肺炎进行数据增强,可能是类别不平衡
            augmented_tensor = do_augmentation(image_tensor)
            augmented_tensor = torch.from_numpy(augmented_tensor)
            augmented_tensor = torch.squeeze(augmented_tensor, dim=0)
            final_tensor = augmented_tensor
        else:
            final_tensor = torch.FloatTensor(image_tensor)
        return final_tensor, labels

    def load_image(self, img_path):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
    
        if self.pre_processing == 'None':
            image = cv2.imread(img_path)
            img_adapteq = Image.fromarray(image.astype('uint8'), 'RGB')
        elif self.pre_processing == 'Equalization':
            image = cv2.imread(img_path)
            image2 = np.copy(image)
            image2[image2>0]=255
            image2 = image2[:,:,0]
            mask = Image.fromarray(image2.astype('uint8'))
            img_adapteq = Image.fromarray(image.astype('uint8'), 'RGB')
            img_adapteq = ImageOps.equalize(img_adapteq,mask=mask)
        elif self.pre_processing == 'CLAHE':
            clahe = cv2.createCLAHE(clipLimit=4,tileGridSize=(8,8))
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
            image[:,:,0] = clahe.apply(image[:,:,0])
            image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)
            img_adapteq = Image.fromarray(image.astype('uint8'), 'RGB')

        preprocess = transforms.Compose([
            transforms.Resize(self.dim[0]),
            transforms.CenterCrop(self.dim[0]),
            transforms.ToTensor(),#normalize to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        image_tensor = preprocess(img_adapteq)
        
        return image_tensor


