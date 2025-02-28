import torch
import os
import random 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
#from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


from network import Encoder, Discriminator
from imageLoader import customDataset




def hdLoss(organ_id_1, disease_id_1,organ_id_2, disease_id_2, h1, h2, epsilon):
    hash_code_length = h1.size(1)
    cos = F.cosine_similarity(h1, h2, dim=1, eps=1e-6)
    #print(cos)
    cos_distH = F.relu((1-cos)*hash_code_length/2)
    #print(cos_distH)
    gt_distH = torch.zeros_like(cos_distH)
    most_similar_indices = (disease_id_1==disease_id_2) #both matched
    similar_indices = (organ_id_1 == organ_id_2) & (disease_id_1 != disease_id_2)#only organ matched
    disimilar_indices = (organ_id_1!=organ_id_2)#no match
    '''print(organ_id_1)
    print(disease_id_1)
    print(organ_id_2)
    print(disease_id_2)'''
    #print(most_similar_indices.sum().item())

    gt_distH[most_similar_indices] = 0
    gt_distH[similar_indices] = hash_code_length/2
    gt_distH[disimilar_indices] = hash_code_length
    #print(gt_distH)
    CHDL = (torch.div(cos_distH-gt_distH.float(), hash_code_length)).cosh().log().sum()

    #print(h1.shape)
    #print(disease_id_1.shape)

    most_similar_cos = cos[most_similar_indices].mean()
    similar_cos = cos[similar_indices].mean()
    no_similar_cos = cos[disimilar_indices].mean()
    #print(similar_cos-most_similar_cos)

    most_similar_dist = (1-most_similar_cos)*hash_code_length/2
    similar_dist = (1-similar_cos)*hash_code_length/2
    no_similar_dist = (1-no_similar_cos)*hash_code_length
    
    #CRL = torch.relu(similar_cos-most_similar_cos)+ torch.relu(no_similar_cos-similar_cos)
    #print(CRL)
    CRL = torch.relu(most_similar_dist-similar_dist+ epsilon)+ torch.relu(similar_dist-no_similar_dist +epsilon)
    return CHDL, CRL, cos_distH, most_similar_indices, similar_indices, disimilar_indices
    
def quantizationLoss(h):
    # Compute the sign of the tensor
    sign_h = torch.sign(h)
    
    # Compute the difference (h_i - sign(h_i))
    diff = h - sign_h
    
    # Compute the log(cosh(diff))
    loss = ((diff.cosh().log())).mean()
    
    # Return the loss
    return loss

# Hyperparameter Details
epochs = 400
hash_code_length = 64
batch_size = 512
learningRate = 0.0001
gamma = 1 #cauchy probablity scale
epsilon = 1 # ranking hyperparemeter



trainingDataPath = "./Train"
#numClasses = len(os.listdir(trainingDataPath))
#print('Num. of classes:', numClasses)

# Model Intilization
encoder = Encoder(hash_code_length)
#classifier = Classifier(numClasses)
discriminator = Discriminator(hash_code_length)
if torch.cuda.is_available():
    encoder.cuda()
    discriminator.cuda()

model_path = os.path.join('./OrganDiseaseRetrieval/Datastore', f'CHDL_quant_{hash_code_length}.pkl')
'''if os.path.exists(model_path):
    encoder.load_state_dict(torch.load(model_path), strict=False)'''

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

trainset = customDataset(trainingDataPath, transform=transform, target_transform=None)
print(len(trainset))
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,  batch_size=batch_size, num_workers=4)

print("\nDataset generated. \n\n")

#loss metrics
#criterion = nn.BCELoss()
#ac1_loss_criterion = nn.NLLLoss()
#discriminator_criterion = nn.BCEWithLogitsLoss()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learningRate, eps=0.0001, amsgrad=True)
#discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = learningRate/10,eps = 0.0001,amsgrad = True)


hd_item = {}
dist_loss_dict = {}
#discriminator_loss_dict = {}
#similar_mean_loss_sict = {}
ranking_loss_dict = {}

temp_loss = 10
for epoch in tqdm(range(epochs)):
    print ("Epoch:%d/%s."%(epoch+1,epochs))
    hd_t0 = 0
    hd_t1 = 0
    hd_t2 = 0
    dist_running_loss =0
    ranking_runing_loss =0
    #d_running_loss =0
    #mean_running_loss = 0
    ranking_count_full = 0
    #ranking_count_full_1 =0
    encoder.train()
    #discriminator.train()
    for i, data in tqdm(enumerate(trainloader, 0)):
        input1, organ_id_1, disease_id_1, index, input2, organ_id_2, disease_id_2 = data #image and label calling

        '''print(organ_id_1)
        print(organ_id_2)
        print(disease_id_1)
        print(disease_id_2)'''
        

        input1, input2 = Variable(input1).cuda(), Variable(input2).cuda()
        organ_id_1, disease_id_1 = Variable(organ_id_1).cuda(), Variable(disease_id_1).cuda()
        organ_id_2, disease_id_2 = Variable(organ_id_2).cuda(), Variable(disease_id_2).cuda()
        #print(input1.shape)
        encoder_optimizer.zero_grad()
        #discriminator_optimizer.zero_grad()
        h1, x1 = encoder(input1)
        h2, x2 = encoder(input2)
        #print(h1.shape)
        #print(x2.shape)
        CHDL, CRL, cos_distH, most_similar_indices, similar_indices, disimilar_indices = hdLoss(organ_id_1, 
                                                        disease_id_1,organ_id_2, disease_id_2, h1, h2, epsilon)
        
        dist_running_loss += CHDL
        ranking_runing_loss += CRL
        #(dist)
        hd_t0 += torch.sum(cos_distH[most_similar_indices]).item()/(cos_distH[most_similar_indices].size(0) + 0.0000001)
        hd_t1 += torch.sum(cos_distH[similar_indices]).item()/(cos_distH[similar_indices].size(0) + 0.0000001)
        hd_t2 += torch.sum(cos_distH[disimilar_indices]).item()/(cos_distH[disimilar_indices].size(0) + 0.0000001)
        
        quant_loss = quantizationLoss(h1)
        total_loss =  CHDL + quant_loss #+CRL
        total_loss.backward()
        #discriminator_optimizer.step()
        encoder_optimizer.step()
        ranking_count_full += 1

        del(input1)
        del(input2)


    dist_loss_dict[epoch] = dist_running_loss.item()/ranking_count_full
    ranking_loss_dict[epoch] = ranking_runing_loss.item()/ranking_count_full
    #similar_mean_loss_sict[epoch] = mean_running_loss.item()/ranking_count_full_1
    #discriminator_loss_dict[epoch] = d_running_loss.item()/ranking_count_full_disciminator
    print('Distnce Loss:', dist_loss_dict[epoch])
    print('Ranking Loss:', ranking_loss_dict[epoch])
    #print('Similar Mean Loss:', similar_mean_loss_sict[epoch])
    #print('Discriminator Loss:', discriminator_loss_dict[epoch])
 

    dataStorePath =  './OrganDiseaseRetrieval/Datastore/'

    CRL_path = os.path.join(dataStorePath, 'Without_CRL_log.pkl')
    with open(CRL_path, 'wb') as handle:
        pickle.dump(ranking_loss_dict, handle)
        print("Saving hamming distance log to ", CRL_path)

    '''hd_item[epoch] = (hd_t0/i, hd_t1/i, hd_t2/i)
    print('hamming distance:', hd_item[epoch])
    hd_path = os.path.join(dataStorePath, 'OPHash_hd_log.pkl')
    with open(hd_path, 'wb') as handle:
        pickle.dump(hd_item, handle)
        print("Saving hamming distance log to ", hd_path)'''

    '''if temp_loss > dist_loss_dict[epoch]:
        temp_loss = dist_loss_dict[epoch]
        encoder_path = os.path.join(dataStorePath, f'CHDL_quant_{hash_code_length}.pkl')
        torch.save(encoder.state_dict(), encoder_path)
        print("Saving model to ", encoder_path)
        print('------------------model is not saved-----------------------')'''
    print('------------------------------------------------------------------------------------------------')
