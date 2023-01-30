#!/usr/bin/env python
# coding: utf-8

# # **Network methods defined basic_blocks**

# In[2]:


#
# https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/blob/master/12_Semantic_Segmentation/Basic_blocks.py
#
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torchvision.utils as v_utils
import torchvision.transforms as transforms
from torch.autograd import Variable


def conv_block(in_dim,out_dim,act_fn,stride,padding):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block(in_dim,out_dim,act_fn,stride,inpadding,outpadding):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=stride, padding=inpadding,output_padding=outpadding),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2(in_dim,out_dim,act_fn,stride,padding):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm2d(out_dim),
    )
    return model    


def conv_block_3(in_dim,out_dim,act_fn,stride,padding):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=stride, padding=padding),
        nn.BatchNorm2d(out_dim),
    )
    return model


# # **Unet Generator**

# In[3]:


#from basic_blocks import * 

class UnetGenerator(nn.Module):

    def __init__(self,in_dim,out_dim,num_filter,dropout=0.5):
        super(UnetGenerator,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        self.dropout = dropout
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        print("\n------Initiating U-Net------\n")

        self.down_1 = conv_block(self.in_dim,self.num_filter,act_fn,stride=1, padding=1)
        self.pool_1 = maxpool()
        self.dropout = nn.Dropout(dropout)
        self.down_2 = conv_block(self.num_filter*1,self.num_filter*2,act_fn,stride=1, padding=1)
        self.pool_2 = maxpool()
        self.down_3 = conv_block(self.num_filter*2,self.num_filter*4,act_fn,stride=1, padding=1)
        self.pool_3 = maxpool()
        self.down_4 = conv_block(self.num_filter*4,self.num_filter*8,act_fn,stride=1, padding=1)
        self.pool_4 = maxpool()

        self.bridge = conv_block(self.num_filter*4,self.num_filter*8,act_fn,stride=1, padding=1)

        self.trans_1 = conv_trans_block(self.num_filter*16,self.num_filter*8,act_fn,stride=2, inpadding=1,outpadding=1)
        self.up_1 = conv_block(self.num_filter*16,self.num_filter*8,act_fn,stride=1, padding=1)
        self.trans_2 = conv_trans_block(self.num_filter*8,self.num_filter*4,act_fn,stride=2, inpadding=1,outpadding=1)
        self.up_2 = conv_block(self.num_filter*8,self.num_filter*4,act_fn,stride=1, padding=1)
        self.trans_3 = conv_trans_block(self.num_filter*4,self.num_filter*2,act_fn,stride=2, inpadding=1,outpadding=1)
        self.up_3 = conv_block(self.num_filter*4,self.num_filter*2,act_fn,stride=1, padding=1)
        self.trans_4 = conv_trans_block(self.num_filter*2,self.num_filter*1,act_fn,stride=2, inpadding=1,outpadding=1)
        self.up_4 = conv_block(self.num_filter*2,self.num_filter*1,act_fn,stride=1, padding=1)

        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter,self.out_dim,3,1,1),
            nn.Tanh(),
        )

    def forward(self,input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(self.dropout(down_1))
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(self.dropout(down_2))
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        #down_4 = self.down_4(pool_3)
        #pool_4 = self.pool_4(down_4)
        bridge = self.bridge(pool_3)

        #trans_1 = self.trans_1(bridge)
        #concat_1 = torch.cat([trans_1,down_4],dim=1)
        #up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(bridge)
        concat_2 = torch.cat([trans_2,down_3],dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3,down_2],dim=1)
        up_3 = self.up_3(concat_3)
        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4,down_1],dim=1)
        up_4 = self.up_4(concat_4)
        out = self.out(up_4)

        return out


# # Fusionnet Generator

# In[4]:


#from Basic_blocks import * 


class Conv_residual_conv(nn.Module):

    def __init__(self,in_dim,out_dim,act_fn):
        super(Conv_residual_conv,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim,self.out_dim,act_fn)
        self.conv_2 = conv_block_3(self.out_dim,self.out_dim,act_fn)
        self.conv_3 = conv_block(self.out_dim,self.out_dim,act_fn)

    def forward(self,input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class FusionGenerator(nn.Module):

    def __init__(self,input_nc, output_nc, ngf):
        super(FusionGenerator,self).__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        print("\n------Initiating FusionNet------\n")

        # encoder

        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # bridge

        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # decoder

        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        # output

        self.out = nn.Conv2d(self.out_dim,self.final_out_dim, kernel_size=3, stride=1, padding=1)
        self.out_2 = nn.Tanh()
        '''
        self.out = nn.Sequential(
            nn.Conv2d(self.out_dim,self.final_out_dim, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(self.final_out_dim),
            nn.Tanh(),
        )
        '''

        # initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self,input):

        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        #down_4 = self.down_4(pool_3)
        #pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_3)

        #deconv_1 = self.deconv_1(bridge)
        #skip_1 = (deconv_1 + down_4)/2
        #up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(bridge)
        skip_2 = (deconv_2 + down_3)/2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2)/2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1)/2
        up_4 = self.up_4(skip_4)

        out = self.out(up_4)
        out = self.out_2(out)
        #out = torch.clamp(out, min=-1, max=1)

        return out


#  

# ==================================================================================================

# # **MAIN**
# 

# ## Set up network

# In[10]:


# Semantic Segmentation
# Code by GunhoChoi
# https://github.com/GunhoChoi/Kind-PyTorch-Tutorial

#from fusionnet import * 
#from unet import *
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torchvision.datasets as dset
import torch.utils.data as data
from torch.nn import MSELoss
#from torch.nn import L1Loss as LSELoss

parser = argparse.ArgumentParser()
parser.add_argument("--network",type=str,default="unet",help="choose between fusionnet & unet")
parser.add_argument("--batch_size",type=int,default=1,help="batch size")
parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
args = parser.parse_args()
num_gpu = args.num_gpu
batch_size = args.batch_size

# hyperparameters
batch_size = 100
lr = 0.0003
train_epochs = 50
num_gpu = 0
network = 'unet'

# input pipeline

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5), (0.5)),
                                transforms.Pad(2)
                               ])
mnist_train = dset.MNIST("./", train=True, transform=transform, target_transform=None, download=True)
mnist_test  = dset.MNIST("./", train=False, transform=transform, target_transform=None, download=True)

# initiate Generator

if network == "fusionnet":
	generator = nn.DataParallel(FusionGenerator(1,1,64),device_ids=[i for i in range(num_gpu)])#.cuda()
elif network == "unet":
	generator = nn.DataParallel(UnetGenerator(1,1,64),device_ids=[i for i in range(num_gpu)])#.cuda()

#

#transforms = transforms.Compose([transforms.Scale(size=img_size), transforms.CenterCrop(size=(img_size,img_size*2))])
# load pretrained model

try:
    generator = torch.load('mkdir{}.pkl'.format(network))
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

# loss function & optimizer

loss_func = MSELoss()
optimizer = torch.optim.Adam(generator.parameters(),lr=lr)

# training


# # Load Data

# In[33]:


print(mnist_train.__len__())
print(mnist_test.__len__())
img1,label1 = mnist_train.__getitem__(0)
img2,label2 = mnist_test.__getitem__(0)

print("img1.size = {}, label1 = {}".format(img1.size(), label1))
print("img2.size = {}, label2 = {}".format(img2.size(), label2))

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=batch_size,shuffle=True)


# ## Train

# In[6]:


# Train Model with train data
# In order to use GPU you need to move all Variables and model by Module.cuda()
from torchvision.transforms.functional import convert_image_dtype
import numpy

earlier_loss = 1000.0
for epoch in range(train_epochs):
    for batch_number,[image,label] in enumerate(train_loader):
        image = Variable(image)#.cuda()
        std_1 = image[:,0,2:5,2:-2].ravel().std()
        std_2 = image[:,0,2:-2,2:5].ravel().std()
        std_img = torch.sqrt(std_1*std_1 + std_2*std_2)#.cuda()
        label = Variable(label)#.cuda()
        #
        optimizer.zero_grad()
        result = generator.forward(image)#.cuda()
        rand_tensor = 2.0*std_img*(torch.rand(image.shape)-0.5)#.cuda()
        #loss = loss_func(result[:,:,2:-2,2:-2],rand_tensor.cuda()[:,:,2:-2,2:-2])
        loss = loss_func(result[:,:,2:-2,2:-2],rand_tensor[:,:,2:-2,2:-2])
        loss.backward()
        optimizer.step()
        #     
        if batch_number % 200 == 0:
            print("\====\nnstandard deviation of image ",std_img)
            print("image shape = {}",format(image.shape))
            print("\nepoch = {} image number = {} loss = {}\n\n".format(epoch,batch_number,loss.detach().cpu().numpy()))
    #
        if loss.detach().cpu().numpy() < earlier_loss:
            numpy.save('./intermediate_output/rand_tensor_{}_{}.npy'.format(epoch,batch_number),rand_tensor.detach().cpu().numpy())
            sd = generator.module.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': sd,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.detach().cpu().numpy(),
            }, "./trained_models/saved_model_epoch_{}_{}.pt".format(epoch,batch_number))
            """
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': sd,
                    'loss': loss.detach().cpu().numpy(),
                    }, './trained_models/saved_model_epoch_{}_{}.pt'.format(epoch,batch_number))
            """
            earlier_loss = loss.detach().cpu().numpy()
#
sd =generator.module.state_dict()
torch.save({
    'epoch': epoch,
    'model_state_dict': sd,
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.detach().cpu().numpy(),
    }, "./trained_models/saved_model_epoch_{}.pt".format(epoch))

"""
torch.save({
            'epoch': epoch,
            'model_state_dict': sd,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.detach().cpu().numpy(),
            }, './saved_final_model_epoch_{}_{}.pt'.format(epoch,batch_number))
"""



# original saved file with DataParallel
#
#unet_restored.load_state_dict(torch.load('./saved_model_epoch_14_363.pt'))
#

state_dict = torch.load('./trained_models/saved_model_epoch_23.pt')
print("\n**\n",state_dict.keys(),"\n**\n")
unet_restored = UnetGenerator(1,1,64)

#unet_restored.load_state_dict(torch.load("./trained_models/saved_model_epoch_23.pt"))
#unet_restored.to(device)


#
#create new OrderedDict that does not contain module.
#
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[0:] # remove module.
    new_state_dict[name] = v
#load params
print("all is well till here \n\n")
unet_restored.load_state_dict(state_dict,strict=False)


#
try:
    device_id = args.device()
except NameError:
    device_id = "cpu"
#
unet_restored.to(device_id)


import numpy
for indx in range(10):
    img_test,label_test = mnist_test.__getitem__(indx)
    out_test = unet_restored(img_test.unsqueeze(0))
    #matplotlib.pyplot.imshow(img_test.squeeze().squeeze().detach().numpy())
    matplotlib.pyplot.show()
    print("\n\n{}\n\n".format(label_test))
    #matplotlib.pyplot.imshow(out_test.squeeze().squeeze().detach().numpy())
    matplotlib.pyplot.show()
    numpy.save('./out_test_{}.npy'.format(23),out_test.detach().cpu().numpy())
    numpy.save('./img_test_{}.npy'.format(23),img_test.detach().cpu().numpy())
    


# In[21]:


numpy.save('./out_test_{}.npy'.format(23),out_test.detach().cpu().numpy())
numpy.save('./img_test_{}.npy'.format(23),img_test.detach().cpu().numpy())


# In[25]:


#matplotlib.pyplot.imshow(out_test.squeeze().squeeze().detach().numpy())



import matplotlib
#matplotlib.pyplot.imshow(img_test.squeeze().squeeze().detach().numpy())

print("\n{}\n".format(label_test))



print(label_test)




