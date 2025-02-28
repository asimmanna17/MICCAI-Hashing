import random
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import os
import PIL.Image as Image
import imageio
from torchvision import  transforms





def sample_return(root):
    newdataset = []
    organ_dict = {'Brain':0, 'Breast':1, 'Chest':2, 'Retina':3}
    disease_dict = {'glioma':0, 'meningioma':1, 'pituitaryTumor':2, 
                    'Bnormal': 3, 'benign':4, 'malignant':5, 
                    'COVID':6, 'ChestNormal':7, 'PNEUMONIA':8, 
                    'NORMAL':9,'CNV':10, 'DME':11,  'DRUSEN':12}
    for path, dirs, files in os.walk(root):
        for file in files:
            img_path = os.path.join(path, file)
            organ_str = file.split('_')[0]
            organ_id = organ_dict[organ_str]
            disease_str = file.split('_')[2]
            disease_id = disease_dict[disease_str]
            item = (img_path, organ_id, disease_id)
            newdataset.append(item)
    return newdataset




class customDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        #classes, class_to_idx = find_classes(root)
        #samples = oversample(root)
        #print(os.path.basename(root))
        # print(len(samples))
        self.root = root
        samples = sample_return(root)
        samples1 = random.sample(samples, len(samples))
        
        
        self.samples = samples
        self.samples1 = samples1

        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        #print('index:',(index))
        path1, organ_id_1, disease_id_1= self.samples[index]
        path2, organ_id_2, disease_id_2= self.samples1[index]
        #print(img)
        #print(img1)
        
        img1 = np.load(path1)
        img2 = np.load(path2)

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        #print(img.shape)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            organ_id_1 = self.target_transform(organ_id_1)
            organ_id_2 = self.target_transform(organ_id_2)
            disease_id_1 = self.target_transform(disease_id_1)
            disease_id_2 = self.target_transform(disease_id_2)
        return img1, organ_id_1, disease_id_1, index, img2, organ_id_2, disease_id_2
    
    def __len__(self):
        return len(self.samples)



#trainpath = "/storage/asim/Group_Hashing_Store/largeSize_updated_dataset/Train/"
'''transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.485), (0.229))
])
trainset = customDataset(trainpath, transform=transform, target_transform=None)
#print(trainset[0])
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,  batch_size=32, num_workers=4)

for i,data1 in (enumerate(trainloader)):
        #print(len(data))
        img1, organ_id_1, disease_id_1, index, img2, organ_id_2, disease_id_2 = data1
        #print(inputs.shape)
        print(organ_id_1)'''
#print(trainset)'''
#print(os.listdir(trainpath))
#newdataset= sample_return(trainpath) 
#print(len(newdataset))
