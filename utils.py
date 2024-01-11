
import json
import os
import shutil
import time

import matplotlib.pyplot as plt
import torch
import torchvision 
from PIL import Image
from torch import optim
from torchvision import transforms,models
from torchvision.models import feature_extraction
import torch.nn as nn
import torch.nn.functional as F

class ImagenetDataset(torch.utils.data.Dataset):
    """
    A tiny version of PASCAL VOC 2007 Detection dataset that includes images and
    annotations with small images and no difficult boxes.
    """

    def __init__(
        self,
        dataset_dir: str,
        motor_fol:str,
        cycle_fol:str,
        image_size: int = 224,
    ):
        """
        Args:
            
            image_size: Size of imges in the batch. The shorter edge of images
                will be resized to this size, followed by a center crop. 
        """
        super().__init__()
        self.image_size = image_size
        self.classes=['motorbike','bicyle']
        
        # Load instances from JSON file:
        motor_dir=os.path.join(dataset_dir,motor_fol)
        bicycle_dir=os.path.join(dataset_dir,cycle_fol)
        
        motor=[os.path.join(motor_dir,file) for file in os.listdir(motor_dir)]
        bicycle=[os.path.join(bicycle_dir,file) for file in os.listdir(bicycle_dir)]
        instances=list()
        for i in range(len(motor)):
            temp={'name':motor[i],'label':1}
            instances.append(temp)
            
        for i in range(len(bicycle)):
            temp={'name':bicycle[i],'label':0}
            instances.append(temp)
            
        self.instances=instances
        self.dataset_dir = dataset_dir

        # Define a transformation function for image: Resize the shorter image
        # edge then take a center crop (optional) and normalize.
        _transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
             transforms.Normalize(
               mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
      
        self.image_transform = transforms.Compose(_transforms)
    def __len__(self):
            return len(self.instances)

    def __getitem__(self, index: int):
        # PIL image and dictionary of annotations.
        instance=self.instances[index]
        image_path, label=instance['name'],instance['label']

        image_path = os.path.join(self.dataset_dir, image_path)
        image = Image.open(image_path).convert("RGB")

        # Transform input image to CHW tensor.
        image = self.image_transform(image)

        
        # Return image path because it is needed for evaluation.
        return image_path, image, label
    
class Classifier(nn.Module):
    def __init__(self,hidden_unit,num_class=1,verbose=True,image_size=224,):
        super().__init__()
        cnn = models.regnet_x_400mf(weights='DEFAULT')
        self.backbone = feature_extraction.create_feature_extractor(
        cnn,
        return_nodes={
            
            "avgpool":'avgpool'
                },
        )
        for child in self.backbone.parameters():
            child.requires_grad=False
        # image_size
        
        dummy=torch.randn(2,3,image_size,image_size)
        out_shape=self.backbone(dummy)['avgpool'].shape
        
#         in_channels=out_shape.shape[1]
        if verbose:
            print('in_channels: ',out_shape)
            
        self.linear=nn.Sequential(nn.Linear(out_shape[1],hidden_unit,bias=True),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5))
        self.cls=nn.Linear(hidden_unit,num_class,bias=True)
    def unfreeze(self):
        for child in self.backbone.parameters():
            child.requires_grad=True
    def forward(self,images):
        x=self.backbone(images)['avgpool']       
        x=x.view(images.shape[0],-1)
        x=self.linear(x)
        x=self.cls(x)                 
        x=F.sigmoid(x)
        return x.squeeze(dim=-1)
    
def train_one_epoch(model,data_loader,optimizer,loss_fn,DEVICE=torch.device('cuda')):
    running_loss = 0.0
    for i, data in enumerate(data_loader):
        _, inputs, labels = data
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass, track history if using gradients for model analysis
        outputs = model(inputs.to(DEVICE))
        loss = loss_fn(outputs.to(float), labels.to(float).to(DEVICE))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print('loss: %.3f' % (running_loss / 10))
            running_loss = 0.0

def accuracy(pred,target,sigmoid=False):
    # if pred>0.5 =>set to 1
    if sigmoid:
        pred=F.sigmoid(pred)
    pred[pred>0.5]=1.
    pred[pred<=0.5]=0.
    acc=(pred==target).nonzero()
    return torch.mean(acc)

def test(img_path):
    img=Image.open(img_path).convert('RGB')
    plt.imshow(img)
    img=torchvision.transforms.functional.pil_to_tensor(img).to(torch.float32)/255.0
    img=torchvision.transforms.functional.resize(img,(224,224))
    img=torchvision.transforms.functional.normalize(img,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img=torch.unsqueeze(img,dim=0)
    pred=model(img.to(DEVICE))
    if pred<0.5:
        print('BICYCLE')
    else:
        print('MOTORBIKE')