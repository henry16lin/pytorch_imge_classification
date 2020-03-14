from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
import os
import numpy as np
from glob import glob
from torchsummary import summary
import network
import time
cwd = os.getcwd()


input_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = 'your/train/imagefolder/path' # for using folder name to know classes
classes = os.listdir(train_path)
model_name = 'customized'
weights_path = os.path.join(cwd,'checkpoint','customized_20200313_163906','epoch-7-val_loss0.000-val_acc1.000.pt')

#test_img_folder = os.path.join(cwd,'testing_data')
test_img_folder = 'test/image/folder'

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
#model = Net(input_size,len(classes)).to(device)
model = network.initialize_model(model_name=model_name,input_size=input_size,
                                     num_classes=len(classes), frozen=True
                                     ,use_pretrained=False,device=device)

network.set_parameter_requires_grad(model,frozen=True )
state = torch.load(weights_path)
model.load_state_dict(state)
summary(model,(3,input_size,input_size))
model.eval()

### do inference ###
pred_result=[]
t0 = time.time()
for img_path in glob(os.path.join(test_img_folder,'*.jpg')):
    t1 = time.time()
    img = image_loader(img_path,data_transforms)
    with torch.no_grad():
        log_prob = F.log_softmax(model(img), dim=1)
        pred_prob = torch.exp(log_prob).data[0].cpu().numpy()

    pred_ind = np.argmax(pred_prob)
    pred_result.append((os.path.basename(img_path),classes[pred_ind],round(pred_prob[pred_ind],3)))

    elapsed = time.time()-t1
    print('predict for %s, result:%s with prob:%.3f, time:%.3fs'
          %(os.path.basename(img_path),classes[pred_ind],pred_prob[pred_ind],elapsed))

total_elapsed = time.time()-t0
print('take %.3fs to inference %d images'%(total_elapsed,len(glob(os.path.join(test_img_folder,'*.jpg')))))

f = open('predict_result.txt','w')
for p in pred_result:
    f.write(str(p)+'\n')
f.close()


