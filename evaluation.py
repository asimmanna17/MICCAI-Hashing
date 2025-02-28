import torch
import os
import numpy as np
import random
import operator
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from torchvision import transforms
from PIL import Image

from network import Encoder
from metric import mAP


random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)



def hammingDistance(h1, h2):
    hash_code = h1.shape[1]
    h1norm = torch.div(h1, torch.norm(h1, p=2))
    h2norm = torch.div(h2, torch.norm(h2, p=2))
    distH = torch.pow(torch.norm(h1norm - h2norm, p=2), 2) * hash_code / 4
    return distH


#### Hyperparemetr Details ######
hash_code_length = 48
#model load######################

model = Encoder(hash_code_length)

if torch.cuda.is_available():
    model.cuda()

model_name = f'OPHash_{hash_code_length}.pkl'
dataStorePath = '/storage/asim/OrganDiseaseRetrieval/Datastore/'
model_path = os.path.join(dataStorePath,model_name)
model.load_state_dict(torch.load(model_path))

print(model_path)
galleryfolderpath = "/storage/asim/Group_Hashing_Store/largeSize_updated_dataset/gallery/"
queryfolderpath = "/storage/asim/Group_Hashing_Store/largeSize_updated_dataset/query/"
gallery_files = os.listdir(galleryfolderpath)
gallery_files = random.sample(gallery_files, len(gallery_files))
query_files = os.listdir(queryfolderpath)
query_files = random.sample(query_files, len(query_files))
print(len(gallery_files))
querynumber = len((query_files))
print(querynumber)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

gallery = {}
print("\n\n Building Gallery .... \n")
model.eval()
with torch.no_grad():
    # Process each gallery image
    for img in gallery_files:
        image_path = os.path.join(galleryfolderpath, img)

        # Load and transform the image
        image = np.load(image_path)
        # transfer to one channel
        if len(image.shape)!= 2:
            image = np.mean(image,axis=-1)

        image = Image.fromarray(image)
        tensor_image = transform(image).unsqueeze(0).cuda()

        # Pass the tensor through the  model
        h, _ = model(tensor_image)

        # Store the result in the gallery dictionary
        gallery[img] = torch.sign(h)

        # Clean up
        del tensor_image
    print("\n Building Complete. \n")

    count = 0

    q_prec_10 = 0
    q_prec_100 = 0
    q_prec_1000 = 0
    
    #print(len(qNimage[0:100]))
    for q_name in query_files:
    
        #
        count = count+1
        query_image_path = os.path.join(queryfolderpath, q_name)
        # Load and transform the image
        query_image = np.load(query_image_path)
        # transfer to one channel
        if len(query_image.shape)!= 2:
            query_image = np.mean(query_image,axis=-1)
        query_image = Image.fromarray(query_image)
        query_tensor_image = transform(query_image).unsqueeze(0).cuda()

        # Pass the tensor through the model
        h_q, _ = model(query_tensor_image)
        h_q = torch.sign(h_q)
        dist = {}
        for key, h1 in gallery.items():
            dist[key] = hammingDistance(h1, h_q)

        print(count)   
        ### images with sorted distance 
        sorted_pool_10 = sorted(dist.items(), key=operator.itemgetter(1))[0:10]
        sorted_pool_100 = sorted(dist.items(), key=operator.itemgetter(1))[0:100]
        sorted_pool_1000 = sorted(dist.items(), key=operator.itemgetter(1))[0:1000]

        #### mean average precision
        q_prec_10 += mAP(q_name, sorted_pool_10)
        q_prec_100 += mAP(q_name, sorted_pool_100)
        q_prec_1000 += mAP(q_name, sorted_pool_1000)

    
        if count % 10 == 0:
            print("mAP@10 :", q_prec_10/count)
            print("mAP@100 :", q_prec_100/count)
            print("mAP@1000 :", q_prec_1000/count)
            


print('-----------------------------------------------')       
print("mAP@10 :", q_prec_10/count)
print("mAP@100 :", q_prec_100/count)
print("mAP@1000 :", q_prec_1000/count)

print(hash_code_length)
print(model_name)

