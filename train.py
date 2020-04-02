# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import numpy as np
import glob
import time
import datetime
from loss.loss import *

########## set parameters ##########
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 10
input_size = 224 #will resize image to input_size*input_size
batch_size = 32
epoch = 50
LR_step = [25,40]
LR_step_ratio = 0.1
num_workers = 0  #on windows pytorch multi-process need to set dataloader inside __name__=='main__'
split_val_by_hand = True  #if true, you need to split validation image into another imagefolder

model_name = 'resnet50' #'customized' for custumized network structure in network.py
use_pretrained = True
frozen = True
#pytorch pre-train model: https://pytorch.org/docs/stable/torchvision/models.html

cwd = os.getcwd()
if split_val_by_hand:
    train_path = 'your/train/imagefolder/path'
    val_path = 'your/validation/imagefolder/path'
    classes = os.listdir(train_path)
else:
    root_path = os.path.join(cwd,'training_data') #training data path
    valid_ratio = 0.2
    classes = os.listdir(root_path)

class_weights = None #given weight list
	
#####################################

now_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
model_save_path = os.path.join(cwd,'checkpoint',model_name+'_'+now_time)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)


# define dataset for dataloader
class mydataset(Dataset):
    def __init__(self, data_path,transform):
        self.transform = transform
        self.data_path = data_path

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self,index):
        imgpath = self.data_path[index]
        img = Image.open(imgpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = classes.index(imgpath.split(os.sep)[-2])
        return img,label


def train(model,device,train_loader,optimizer,epoch,class_weights):
    print('train on %d data......'%len(train_loader.dataset))
    train_loss = 0
    correct = 0
    model.train()
    for batch_ind,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        weights =  torch.tensor(class_weights)
        #criterion = FocalLoss(weight=weights)
        criterion = nn.CrossEntropyLoss(weight=weights)
        loss = criterion(output, target)

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

        #if batch_ind %30==0: #print during batch runing
        #    print('Train Epoch: %d [%d/%d](%.2f%%)\tLoss:%.6f' %(epoch, batch_ind*len(data),
        #          len(train_loader.dataset),100.*batch_ind/len(train_loader),loss.item() ))

    train_loss /= len(train_loader.dataset)
    acc = correct/len(train_loader.dataset)

    print('Train epoch:%d, average loss:%.4f, acc:%d/%d(%.3f%%)' %(epoch,train_loss,
          correct, len(train_loader.dataset), 100*acc))

    return train_loss, acc


def val(model,device,val_loader,class_weights):
    print('validation on %d data......'%len(val_loader.dataset))
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad(): #temporarily set all the requires_grad flag to false
        for data,target in val_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            
            weights =  torch.tensor(class_weights)
            #criterion = FocalLoss(weight=weights)
            criterion = nn.CrossEntropyLoss(weight=weights)
            val_loss += criterion(output, target).item() #sum up batch loss

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        val_loss /= len(val_loader.dataset) #avg of sum of batch loss

    print('Val set:Average loss:%.4f, acc:%d/%d(%.3f%%)' %(val_loss,
          correct, len(val_loader.dataset), 100.*correct/len(val_loader.dataset)))

    return val_loss, correct/len(val_loader.dataset)




if __name__=='__main__':

    data_transforms = {
        'train':
        transforms.Compose([transforms.Resize((input_size,input_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225]) ]),
        'validation':
        transforms.Compose([transforms.Resize((input_size,input_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225]) ])
        }


    if not split_val_by_hand:
        ### prepare for train,val,test set
        data_path=[]
        for c in classes:
            data_path += glob.glob(os.path.join(root_path,c,'*.jpg'))

        #split val from all train data
        data_size = len(data_path)
        indices = list(range(data_size))
        split = int(np.floor(valid_ratio * data_size))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_path = [data_path[i] for i in train_indices]
        val_path = [data_path[i] for i in val_indices]
        
        # use custom dataset
        train_data = mydataset(train_path,data_transforms['train'])
        val_data = mydataset(val_path,data_transforms['validation'])
        # or use ImageFolder to create dataset and random_split to separate train and validation set
        # but train and val can't use different transform
        #from torch.utils.data.dataset import random_split
        #torch.manual_seed(random_seed)
        #init_dataset = torchvision.datasets.ImageFolder(root_path,data_transforms)
        #print(init_dataset.class_to_idx)
        #lengths = [int(len(init_dataset)*(valid_ratio)), int(len(init_dataset)*valid_ratio)]
        #train_data, val_data = random_split(init_dataset, lengths)

    else:
        train_data = torchvision.datasets.ImageFolder(train_path,data_transforms['train'])
        val_data = torchvision.datasets.ImageFolder(val_path,data_transforms['validation'])




    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               pin_memory=True,num_workers=num_workers,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                             pin_memory=True,num_workers=num_workers,shuffle=True)

    '''
    ## see some image ##
    import matplotlib.pyplot as plt
    mean,std = torch.tensor([0.485, 0.456, 0.406]),torch.tensor([0.229, 0.224, 0.225])

    def show_image(image):
        image = transforms.Normalize(-mean/std,1/std)(image)
        np_image = image.numpy()
        plt.imshow(np.transpose(np_image,(1,2,0)))

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    show_image(torchvision.utils.make_grid(images,3))
    print([classes[i] for i in labels])
    '''

    ### creat model ###
    from torchsummary import summary
    import network


    model = network.initialize_model(model_name=model_name,input_size=input_size,
                                     num_classes=len(classes), frozen=frozen
                                     ,use_pretrained=use_pretrained,device=device)
    print(summary(model,(3,input_size,input_size)))
    #model.parameters

    optimizer = optim.Adam(model.parameters())
    #scheduler = StepLR(optimizer, step_size=LR_step_size, gamma=LR_step_ratio)
    scheduler = MultiStepLR(optimizer, milestones=LR_step, gamma=LR_step_ratio, last_epoch=-1)

    ### training ###
    val_loss = 10000000
    val_acc = 0
    train_loss_hist,train_acc_hist = [],[]
    val_loss_hist,val_acc_hist = [],[]
    for ep in range(1, epoch + 1):
        epoch_begin = time.time()
        cur_train_loss,cur_train_acc = train(model,device,train_loader,optimizer,ep,class_weights)
        cur_val_loss,cur_val_acc = val(model,device,val_loader,class_weights)
        scheduler.step()
        print('elapse:%.2fs \n'%(time.time()-epoch_begin))

        if cur_val_loss<=val_loss:
            print('improve validataion loss, saving model...\n')
            torch.save(model.state_dict(), os.path.join(model_save_path,
                       'epoch-%d-val_loss%.3f-val_acc%.3f.pt' %(ep,cur_val_loss,cur_val_acc) ))
            val_loss = cur_val_loss
            val_acc = cur_val_acc

        train_loss_hist.append(cur_train_loss)
        train_acc_hist.append(cur_train_acc)
        val_loss_hist.append(cur_val_loss)
        val_acc_hist.append(cur_val_acc)


    #save final model
    state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
    torch.save(state, os.path.join(model_save_path,'last_model.pt'))


    ### graph train hist ###
    import matplotlib.pyplot as plt
    #plt.style.use('ggplot')

    fig = plt.figure()
    plt.plot(train_loss_hist)
    plt.plot(val_loss_hist)
    plt.legend(['train loss','val loss'],loc='best')
    plt.savefig(os.path.join(model_save_path,'loss.jpg'))
    plt.close(fig)
    fig = plt.figure()
    plt.plot(train_acc_hist)
    plt.plot(val_acc_hist)
    plt.legend(['train acc','val acc'],loc='best')
    plt.savefig(os.path.join(model_save_path,'acc.jpg'))
    plt.close(fig)





