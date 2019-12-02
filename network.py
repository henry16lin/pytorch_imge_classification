# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    c_script = [64,64,'p',128,128,'p',256,256,256,'p',128,512,128,512,128,512,'p',128,512,128,512,128,512]
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
        x = self.classify(x)
        output = F.log_softmax(x, dim=1)

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


