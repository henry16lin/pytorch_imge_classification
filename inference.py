from PIL import Image
from torchvision import transforms
import torch
import os
import numpy as np
from glob import glob
from torchsummary import summary
from network import Net

input_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['cat','dog']
model_name = 'epoch-1-val_loss0.147-val_acc0.546.pt'
cwd = os.getcwd()
test_img_folder = os.path.join(cwd,'testing_data')

data_transforms = transforms.Compose([transforms.Resize((input_size,input_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225]) ])

def image_loader(image,transform):
    image = Image.open(image)
    image = data_transforms(image)
    image = image.unsqueeze(0)

    return image.to(device)


### load model and model weight ###
model = Net(input_size,len(classes)).to(device)
state = torch.load(os.path.join(cwd,'checkpoint',model_name))
model.load_state_dict(state)
summary(model,(3,input_size,input_size))
model.eval()

### do inference ###
pred_result=[]
for img_path in glob(os.path.join(test_img_folder,'*.jpg')):
    img = image_loader(img_path,data_transforms)
    print('predict for %s'%os.path.basename(img_path))
    with torch.no_grad():
        pred_prob = torch.exp(model(img)).data[0].cpu().numpy()

    pred_ind = np.argmax(pred_prob)
    pred_result.append((os.path.basename(img_path),classes[pred_ind],round(pred_prob[pred_ind],3)))



f = open('predict_result.txt','w')
for p in pred_result:
    f.write(str(p)+'\n')
f.close()


