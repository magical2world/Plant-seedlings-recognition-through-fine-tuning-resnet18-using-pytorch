import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from PIL import Image

'''
Setting parameters
'''
n_classes=12
use_gpu=True
image_size=224
folder_path='train'
'''
build dataset for training
'''
def load_img(path):
   img=Image.open(path)
   img=img.convert("RGB")
   img=img.resize((224,224))
   img=np.array(img)/255.0
   return img
images=[]
labels=[]
image_names=[]
for idx,image_folder in enumerate(os.listdir(folder_path)):
    for filename in os.listdir(folder_path+'/'+image_folder):
       labels.append(idx)
       image_names.append(image_folder)
       images.append(load_img(folder_path+'/'+image_folder+'/'+filename))
image_dict={}
for i in range(len(labels)):
    image_dict[labels[i]]=image_names[i]
images=np.array(images)
images=np.transpose(images,[0,3,1,2])
labels=np.array(labels)
labels_tensor=torch.from_numpy(labels).float()
images_tensor=torch.from_numpy(images).float()
torch_dataset=torch.utils.data.TensorDataset(data_tensor=images_tensor,target_tensor=labels_tensor)
dataloader=torch.utils.data.DataLoader(torch_dataset,batch_size=128,shuffle=True)
'''
build model
'''
resnet=torchvision.models.resnet18(pretrained=True)
num_ftrs=resnet.fc.in_features
resnet.fc=nn.Linear(num_ftrs,n_classes)
'''
Optimizer
'''
criterion=nn.CrossEntropyLoss()
if use_gpu:
   criterion.cuda()
   resnet.cuda()

optimizer=optim.RMSprop(resnet.parameters(),lr=0.0001)

def accuracy(predict,target):
    predict=predict.data.cpu().numpy()
    target=target.data.cpu().numpy()
    correct=np.mean(np.equal(np.argmax(predict,1),target))
    return correct
'''
Train
'''
for epoch in range(50):
    for idx,(data,target) in enumerate(dataloader):
        data,target=Variable(data),Variable(target.type(torch.LongTensor))
        if use_gpu:
            data,target=data.cuda(),target.cuda()
        optimizer.zero_grad()
        predict=resnet(data)
        loss=criterion(predict,target)
        loss.backward()
        optimizer.step()
        if idx%10==0:
            print('epoch:%d,step:%d,loss:%f'%(epoch,idx,loss.data[0]))
            print('epoch:%d,step:%d,accuracy:%f'%(epoch,idx,accuracy(predict,target)))



