import random
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

from micro import TEST, TRAIN, VAL


class ISIC2018_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=os.getcwd()
        self.mode=mode
        gts_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1_Training_GroundTruth','ISIC2018_Task1_Training_GroundTruth')
        images_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1-2_Training_Input','ISIC2018_Task1-2_Training_Input')
        
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]
        self.data=[]
        for i in range(len(images_list)):
            image_path=images_path+'/'+images_list[i]
            mask_path=gts_path+'/'+gts_list[i]
            self.data.append([image_path, mask_path])
        self.transformer=transformer
        random.shuffle(self.data)

        
        if mode==TRAIN:
            self.data=self.data[:1815]
        elif mode==VAL:
            self.data=self.data[1815:2074]
        elif mode==TEST:
            self.data=self.data[2074:2594]
        print(len(self.data))
        self.data_buf=self.cuda_buffer()
        
    def cuda_buffer(self):
        data_buf=[]
        id=0
        for data in self.data:
            image_path,gt_path=data
            image = Image.open(image_path).convert('RGB')
            image=np.array(image)
            image = np.transpose(image, axes=(2, 0, 1))
            gt = Image.open(gt_path).convert('L')
            gt = np.array(gt)
            gt=np.expand_dims(gt, axis=2) / 255
            gt = np.transpose(gt, axes=(2, 0, 1))
            image, gt = self.transformer((image, gt))
            image=image.cuda()
            gt=gt.cuda()
            if self.mode==TEST:
                data_buf.append([image,gt,image_path.split('/')[-1]])
            else:
                data_buf.append([image,gt])
            if id%20==0:
                print(id)
            id=id+1
        return data_buf
    
    def __getitem__(self, index):
        if self.mode!=TEST:
            image, gt=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt
        else:
            image, gt,image_name=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt,image_name     

    def __len__(self):
        return len(self.data)



class ISIC2017_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=os.getcwd()
        self.mode=mode
        if mode==TRAIN:
            gts_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Training_Part1_GroundTruth','ISIC-2017_Training_Part1_GroundTruth')
            images_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Training_Data','ISIC-2017_Training_Data')
        elif mode==VAL:
            gts_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Validation_Part1_GroundTruth','ISIC-2017_Validation_Part1_GroundTruth')
            images_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Validation_Data','ISIC-2017_Validation_Data')
        elif mode==TEST:
            gts_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Test_v2_Part1_GroundTruth','ISIC-2017_Test_v2_Part1_GroundTruth')
            images_path=os.path.join(cwd,'data','ISIC2017','ISIC-2017_Test_v2_Data','ISIC-2017_Test_v2_Data')
        
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "superpixels" not in item]
        gts_list=sorted(os.listdir(gts_path))
        self.data=[]
        for i in range(len(images_list)):
            image_path=images_path+'/'+images_list[i]
            mask_path=gts_path+'/'+gts_list[i]
            self.data.append([image_path, mask_path])
        random.shuffle(self.data)
        self.transformer=transformer
        self.data_buf=self.cuda_buffer()
        
  
    def cuda_buffer(self):
        data_buf=[]
        id=0
        for data in self.data:
            image_path,gt_path=data
            image = Image.open(image_path).convert('RGB')
            image=np.array(image)
            image = np.transpose(image, axes=(2, 0, 1))
            gt = Image.open(gt_path).convert('L')
            gt = np.array(gt)
            gt=np.expand_dims(gt, axis=2) / 255
            gt = np.transpose(gt, axes=(2, 0, 1))
            image, gt = self.transformer((image, gt))
            image=image.cuda()
            gt=gt.cuda()
            if self.mode==TEST:
                data_buf.append([image,gt,image_path.split('/')[-1]])
            else:
                data_buf.append([image,gt])
            if id%20==0:
                print(id)
            id=id+1
        return data_buf
    
    def __getitem__(self, index):
        if self.mode!=TEST:
            image, gt=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt
        else:
            image, gt,image_name=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt,image_name     

    def __len__(self):
        return len(self.data)



class PH2_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        self.mode=mode
        cwd=os.getcwd()
        images_path=os.path.join(cwd,'data','PH2','PH2Dataset','PH2 Dataset images')
        images_list=sorted(os.listdir(images_path))
        random.shuffle(images_list)
        self.data=[]
        for path in images_list:
            image_path=os.path.join(images_path,path,path+'_Dermoscopic_Image',path+'.bmp')
            gt_path=os.path.join(images_path,path,path+'_lesion',path+'_lesion.bmp')
            self.data.append([image_path, gt_path])
        if mode==TRAIN:
            self.data=self.data[:160]
        if mode==VAL:
            self.data=self.data[160:180]
        if mode==TEST:
            self.data=self.data[180:200]
        self.transformer=transformer
        print(f'the length of datasets is {len(self.data)}')
    
    def __getitem__(self, index):
        image_path, gt_path=self.data[index]
        image = Image.open(image_path).convert('RGB')
        image=np.array(image)
        image = np.transpose(image, axes=(2, 0, 1))
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt=np.expand_dims(gt, axis=2) / 255
        gt = np.transpose(gt, axes=(2, 0, 1))
        image, gt = self.transformer((image, gt))
        if self.mode==TEST:
            return image,gt,image_path.split('/')[-1]
        return image,gt

    def __len__(self):
        return len(self.data)

class BUSI_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=os.getcwd()
        self.mode=mode
        data_path_1=os.path.join(cwd,'data','BUSI','Dataset_BUSI','Dataset_BUSI_with_GT','benign')
        data_path_2=os.path.join(cwd,'data','BUSI','Dataset_BUSI','Dataset_BUSI_with_GT','malignant')
        benign_list=sorted(os.listdir(data_path_1))
        malignant_list=sorted(os.listdir(data_path_2))

        benign_image_list=[item for item in benign_list if ").png" in item]
        benign_gt_list=[item for item in benign_list if "mask.png" in item]

        malignant_image_list=[item for item in malignant_list if ").png" in item]
        malignant_gt_list=[item for item in malignant_list if "mask.png" in item]

        
        self.data=[]
        for i in range(len(benign_image_list)):
            image_path=data_path_1+'/'+benign_image_list[i]
            mask_path=data_path_1+'/'+benign_gt_list[i]
            self.data.append([image_path, mask_path])
        for i in range(len(malignant_image_list)):
            image_path=data_path_2+'/'+malignant_image_list[i]
            mask_path=data_path_2+'/'+malignant_gt_list[i]
            self.data.append([image_path, mask_path])

        random.shuffle(self.data)

        if mode==TRAIN:
            self.data=self.data[:518]
        if mode==VAL:
            self.data=self.data[518:647]
        if mode==TEST:
            self.data=self.data[518:647]
        
        self.transformer=transformer
        print(len(self.data))
        self.data_buf=self.cuda_buffer()


    def cuda_buffer(self):
        data_buf=[]
        id=0
        for data in self.data:
            image_path,gt_path=data
            image = Image.open(image_path).convert('RGB')
            image=np.array(image)
            image = np.transpose(image, axes=(2, 0, 1))
            gt = Image.open(gt_path).convert('L')
            gt = np.array(gt)
            gt=np.expand_dims(gt, axis=2) / 255
            gt = np.transpose(gt, axes=(2, 0, 1))
            image, gt = self.transformer((image, gt))
            image=image.cuda()
            gt=gt.cuda()
            if self.mode==TEST:
                data_buf.append([image,gt,image_path.split('/')[-1]])
            else:
                data_buf.append([image,gt])
            if id%20==0:
                print(id)
            id=id+1
        return data_buf
    
    def __getitem__(self, index):
        if self.mode!=TEST:
            image, gt=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt
        else:
            image, gt,image_name=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt,image_name     

    def __len__(self):
        return len(self.data)




class Kvasir_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=os.getcwd()
        self.mode=mode
        if mode==TRAIN:
            gts_path=os.path.join(cwd,'data','Kvasir','kvasir-seg','Kvasir-SEG','masks')
            images_path=os.path.join(cwd,'data','Kvasir','kvasir-seg','Kvasir-SEG','images')
        elif mode==VAL:
            gts_path=os.path.join(cwd,'data','Kvasir','kvasir-seg','Kvasir-SEG','masks')
            images_path=os.path.join(cwd,'data','Kvasir','kvasir-seg','Kvasir-SEG','images')
        elif mode==TEST:
            gts_path=os.path.join(cwd,'data','Kvasir','kvasir-seg','Kvasir-SEG','masks')
            images_path=os.path.join(cwd,'data','Kvasir','kvasir-seg','Kvasir-SEG','images')

        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "jpg" in item]
        self.data=[]
        for i in range(len(images_list)):
            image_path=images_path+'/'+images_list[i]
            mask_path=gts_path+'/'+gts_list[i]
            self.data.append([image_path, mask_path])
        self.transformer=transformer
        random.shuffle(self.data)
        if mode==TRAIN:
            self.data=self.data[:880]
        elif mode==VAL:
            self.data=self.data[880:1000]
        elif mode==TEST:
            self.data=self.data[880:1000]

        print(len(self.data))
        self.data_buf=self.cuda_buffer()
        
    def cuda_buffer(self):
        data_buf=[]
        id=0
        for data in self.data:
            image_path,gt_path=data
            image = Image.open(image_path).convert('RGB')
            image=np.array(image)
            image = np.transpose(image, axes=(2, 0, 1))
            gt = Image.open(gt_path).convert('L')
            gt = np.array(gt)
            gt=np.expand_dims(gt, axis=2) / 255
            gt = np.transpose(gt, axes=(2, 0, 1))
            image, gt = self.transformer((image, gt))
            image=image.cuda()
            gt=gt.cuda()
            if self.mode==TEST:
                data_buf.append([image,gt,image_path.split('/')[-1]])
            else:
                data_buf.append([image,gt])
            if id%20==0:
                print(id)
            id=id+1
        return data_buf
    
    def __getitem__(self, index):
        if self.mode!=TEST:
            image, gt=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt
        else:
            image, gt,image_name=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt,image_name

    def __len__(self):
        return len(self.data)