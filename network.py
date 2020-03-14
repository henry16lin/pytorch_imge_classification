# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


def make_classifier(in_features, mid_features, num_classes):
    return nn.Sequential(
        nn.Linear(in_features, mid_features),
        nn.BatchNorm1d(num_features=mid_features),
        nn.LeakyReLU(0.1,inplace=True),
        nn.Linear(mid_features, num_classes)
    )


def set_parameter_requires_grad(model, frozen):
    if frozen:
        for param in model.parameters():
            param.requires_grad = False



def initialize_model(model_name,input_size, num_classes, frozen, use_pretrained, device):


    if model_name == 'customized':
        model = Net(input_size,num_classes).to(device)

    else:

        try:
            net = getattr(models,model_name)(pretrained=use_pretrained)
        except:
            print('there is no pre-train model named: %s'%model_name)
            print('pre-train list can see from https://pytorch.org/docs/stable/torchvision/models.html')
            raise AttributeError()


        model = net.to(device)
        set_parameter_requires_grad(model, frozen=frozen)

        if model_name.startswith('vgg'):
            #model.classifier[6] = nn.Linear(in_features= model.classifier[6].in_features, out_features=num_classes, bias=True).to(device)
            model.classifier = make_classifier(model.classifier[0].in_features,512,num_classes).to(device)

        elif model_name.startswith('resnet') or model_name.startswith('inception'):
            model.fc = make_classifier(model.fc.in_features,512,num_classes).to(device)

        elif model_name.startswith('densenet'):
            model.classifier = make_classifier(model.classifier.in_features,512,num_classes).to(device)

        elif model_name.startswith('mobilenet'):
            model.classifier[1] = nn.Linear(in_features= model.classifier[1].in_features, out_features=num_classes, bias=True).to(device)

        else:
            raise Exception('please update classifier by in network.py!')

    return model


class Net(nn.Module):
    #c_script = [64,64,'p',128,128,'p',256,256,256,'p',512,512,512,'p',512,512,512]
    #k_script = [3,3,2, 3,3,2, 3,3,3,2,  3,3,3,2  ,3,3,3]
    c_script = [32,32,'p',64,64,'p',128,128,128,'p',32,128,32,128,32,128,'p',64,256,64,256,64,256]
    k_script = [3,3,2, 3,3,2, 3,3,3,2,  1,3,1,3,1,3,2, 1,3,1,3,1,3]


    def __init__(self,input_size,num_class):
        super(Net, self).__init__()
        self.conv_feature = self.make_layer()

        self.gap_size = int(np.floor(input_size/2**sum([i=='p' for i in self.c_script])))
        self.gap = nn.AvgPool2d(self.gap_size)

        self.classify = nn.Linear(self.c_script[-1],num_class,bias=False)

    def forward(self,x):
        x = self.conv_feature(x)
        x = self.gap(x)
        x = x.view(-1,self.c_script[-1])
        output = self.classify(x)
        #output = F.log_softmax(x, dim=1)

        return output

    def make_layer(self,in_channels=3):
        c_script,k_script = self.c_script,self.k_script
        layers = []
        for i in range(len(c_script)):
            if c_script[i]=='p':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]

            else:
                pad = 0 if k_script[i]==1 else 1
                conv2d = nn.Conv2d(in_channels,c_script[i],kernel_size=k_script[i],padding=pad)
                layers+=[conv2d,nn.BatchNorm2d(c_script[i]),nn.LeakyReLU(0.1,inplace=True)]

                in_channels = c_script[i]

        return nn.Sequential(*layers)


'''
t_data = torch.rand([9,3,127,127])
cnn = make_layer(c_script,k_script,3)
t_result = cnn(t_data)
t_result.size()
'''


