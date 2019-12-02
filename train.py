import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import numpy as np
import glob
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
valid_ratio = 0.25
random_seed = 10
input_size = 128 #will resize image to input_size*input_size
batch_size = 5
epoch = 100
num_workers = 0 #on windows pytorch multi-process need to set dataloader inside __name__=='main__'

cwd = os.getcwd()
root_path = os.path.join(cwd,'training_data') #training data root path
classes = os.listdir(root_path)


cwd = os.getcwd()
model_save_path = os.path.join(cwd,'checkpoint')
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# define dataset for dataloader <--equivalent to torchvision.datasets.ImageFolder
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

        label = classes.index(imgpath.split('\\')[-2])
        
        return img,label


def train(model,device,train_loader,optimizer,epoch,classes_weight=None):
    train_loss = 0
    correct = 0
    model.train()
    for batch_ind,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target,classes_weight)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

        if batch_ind %30==0:
            print('Train Epoch: %d [%d/%d](%.2f%%)\tLoss:%.6f' %(epoch, batch_ind*len(data),
                  len(train_loader.dataset),100.*batch_ind/len(train_loader),loss.item() ))

    train_loss /= len(train_loader.dataset)
    return train_loss, correct/len(train_loader.dataset)


def val(model,device,val_loader):
    print('validation on %d data......'%len(val_loader.dataset))
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad(): #temporarily set all the requires_grad flag to false
        for data,target in val_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output,target).item() #sum up batch loss

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        val_loss /= len(val_loader.dataset) #avg of sum of batch loss

    print('Val set:Average loss:%.4f, acc:%d/%d(%.3f%%)' %(val_loss,
          correct, len(val_loader.dataset), 100.*correct/len(val_loader.dataset)))

    return val_loss, correct/len(val_loader.dataset)


if __name__=='__main__':

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

    ### check classes balance in training set ###
    train_class_cnt = np.zeros(len(classes))
    for i in range(len(train_path)):
        train_class_cnt[classes.index(train_path[i].split('\\')[-2])]+=1
    
    val_class_cnt = np.zeros(len(classes))
    for i in range(len(val_path)):
        val_class_cnt[classes.index(val_path[i].split('\\')[-2])]+=1
    print('training set class count: '+str(train_class_cnt))
    print('validation set class count: '+str(val_class_cnt))
    
    if np.any(train_class_cnt/sum(train_class_cnt)<0.2):
        classes_weight = 1-(train_class_cnt/sum(train_class_cnt))
        classes_weight = torch.tensor(list(classes_weight))
        print('size imbalance! will weight class in loss')
    else:
        classes_weight = None
    
    
    
    data_transforms = transforms.Compose([transforms.Resize((input_size,input_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225]) ])

    # use custom dataset
    train_data = mydataset(train_path,data_transforms)
    val_data = mydataset(val_path,data_transforms)

    '''
    # or use ImageFolder to create dataset and random_split to separate train and validation set
    from torch.utils.data.dataset import random_split
    torch.manual_seed(random_seed)
    init_dataset = torchvision.datasets.ImageFolder(root_path,data_transforms)
    print(init_dataset.class_to_idx)
    lengths = [int(len(init_dataset)*0.8), int(len(init_dataset)*0.2)]
    train_data, val_data = random_split(init_dataset, lengths)
    '''


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               pin_memory=True,num_workers=num_workers,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                             pin_memory=True,num_workers=num_workers,shuffle=True)

    
    #see some image
    mean,std = torch.tensor([0.485, 0.456, 0.406]),torch.tensor([0.229, 0.224, 0.225])

    def show_image(image):
        image = transforms.Normalize(-mean/std,1/std)(image)
        np_image = image.numpy()
        plt.imshow(np.transpose(np_image,(1,2,0)))

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    show_image(torchvision.utils.make_grid(images,3))
    print([classes[i] for i in labels])
    

    ### creat model structure ###
    from torchsummary import summary
    from network import Net

    model = Net(input_size,len(classes)).to(device)
    summary(model,(3,input_size,input_size))
    #model.parameters

    optimizer = optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    ### training ###
    val_loss = 10000000
    val_acc = 0
    train_loss_hist,train_acc_hist = [],[]
    val_loss_hist,val_acc_hist = [],[]
    print('train on %d data with %d classes......'%(len(train_loader.dataset),len(classes) ))
    for ep in range(1, epoch + 1):
        epoch_begin = time.time()
        cur_train_loss,cur_train_acc = train(model,device,train_loader,optimizer,ep,classes_weight)
        cur_val_loss,cur_val_acc = val(model,device,val_loader)
        scheduler.step()
        print('elapse:%.2f seconds \n'%(time.time()-epoch_begin))

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
    fig = plt.figure()
    plt.plot(train_loss_hist)
    plt.plot(val_loss_hist)
    plt.legend(['train loss','val loss'],loc='best')
    plt.savefig('loss.jpg')
    plt.close(fig)
    fig = plt.figure()
    plt.plot(train_acc_hist)
    plt.plot(val_acc_hist)
    plt.legend(['train acc','val acc'],loc='best')
    plt.savefig('acc.jpg')
    plt.close(fig)





