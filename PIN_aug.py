# -*- coding: utf-8 -*-
import pickle
import os
import clip
import torch
import network
import torch.nn as nn
import torch.nn.functional as F
from utils.stats import calc_mean_std
import argparse
from main import get_dataset
from torch.utils import data
import numpy as np
import random


# +
# For target domain image feature

class PIN(nn.Module):
    def __init__(self,shape,content_feat):
        super(PIN,self).__init__()
        self.shape = shape
        self.content_feat = content_feat.clone().detach()
        self.content_mean, self.content_std = calc_mean_std(self.content_feat)
        self.size = self.content_feat.size()
        self.content_feat_norm = (self.content_feat - self.content_mean.expand(
        self.size)) / self.content_std.expand(self.size)

        self.style_mean = self.content_mean.clone().detach() 
        self.style_std  = self.content_std.clone().detach()

        self.style_mean = nn.Parameter(self.style_mean, requires_grad = True)
        self.style_std  = nn.Parameter(self.style_std, requires_grad = True)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self):
        self.style_std.data.clamp_(min=0)
        
        target_feat =  self.content_feat_norm \
                        * self.style_std.expand(self.size) \
                        + self.style_mean.expand(self.size)
        
        target_feat = self.relu(target_feat)
        return target_feat


# -

available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and not (name.startswith("__") or name.startswith('_')) and callable(network.modeling.__dict__[name]))

gpu_id = "0"
data_root = "/workspace/dataset/GTAV"
save_dir = "exp2"
dataset = "gta5"
crop_size = 768
batch_size = 80
model = "deeplabv3plus_resnet_clip"
BB = "RN50"
weight_decay = 1e-4
total_it = 100
resize_feat = True
random_seed = 123

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

train_dst, val_dst = get_dataset(dataset, data_root, crop_size, data_aug=False)
train_loader = data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

print("Dataset: %s, Train set: %d, Val set: %d" % (dataset, len(train_dst), len(val_dst)))

cur_itrs = 0

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
if resize_feat:
    t1 = nn.AdaptiveAvgPool2d((56, 56))
else:
    t1 = lambda x: x

import clip
from clip.simple_tokenizer import *


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


# +
device = 'cuda:0'
clip_model, preprocess = clip.load('RN50', device, jit=False)

text_encoder = TextEncoder(clip_model)
# tokenizer = SimpleTokenizer()
# -

def diversity_loss(embedding_list, embedding, idx):
    Lstyle = 0.0
        
    for i in range(idx):
        prev_embedding = embedding_list[i]
        
        prev_embedding = prev_embedding / prev_embedding.norm(dim=-1, keepdim=True)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        Lstyle = Lstyle + abs(embedding @ prev_embedding.T)
    
    Lstyle = Lstyle / idx
    
    return Lstyle


# +
# training iteration -> argument
L = 100
# number of style vectors -> argument
K = 80
# dimension
D = 512

style_feature_list =[]

n_ctx = 1
Lstyle = 0
# -

for j in range(K):
    ctx_vectors = torch.empty(1, 1, D)
    nn.init.normal_(ctx_vectors, std=0.2)
    # ctx_vectors = ctx_vectors.repeat(7, 1, 1)
    prompt_prefix = " ".join(["X"] * n_ctx)
    
    # optimize vector
    ctx = nn.Parameter(ctx_vectors)
    
    # optimizer & scheduler 
    optimizer = torch.optim.SGD([ctx], lr=0.01, momentum=0.9)
        
    ctx_half = 'A diverse driving conditions in'
    
    # classnames = [name.replace("_", " ") for name in class_name]
    
    for x in range(L):
        prompts = [ctx_half + " " + prompt_prefix]
        tokenized_prompts = clip.tokenize(prompts).to(device)
        embedding = clip_model.token_embedding(tokenized_prompts)
        
        prefix = embedding[:, :6, :].to(device)
        suffix = embedding[:, 6+n_ctx :, :].to(device)
        
        # "<SOS> A diverse driving conditions at X"라는 임베딩 값을 "<SOS> Driving conditions at S*"로 변경
        ctx_i = ctx[:, :, :].to(device)
        prefix_i = prefix[:, :, :].to(device)
        suffix_i = suffix[:, :, :].to(device)
        
        prompts = torch.cat((prefix_i, ctx_i, suffix_i), dim=1).half()
        
        # style_feature : "Driving conditions at S*"
        style_feature = text_encoder(prompts.half(), tokenized_prompts.long())
        
        # style_feature = style_feature / style_feature.norm(dim=-1, keepdim=True)
        
        if j == 0:
            break
        
        # Lstyle loss
        Lstyle = diversity_loss(style_feature_list, style_feature, j)
        
        optimizer.zero_grad()
        Lstyle.backward()
        optimizer.step()
        
    if j != 0:
        print(f"{j+1} : loss is {Lstyle.item():4f}")

    style_feature_list.append(style_feature.detach())

style_feature_list = torch.stack(style_feature_list)
style_feature_list = style_feature_list.squeeze(1)

with open('style_feature.pkl', 'wb') as f:
    pickle.dump(style_feature_list, f)

style_feature_list = torch.stack(style_feature_list)
style_feature_list = style_feature_list.squeeze(1)

model = network.modeling.__dict__["deeplabv3plus_resnet_clip"](num_classes=19, 
                                                               BB=BB, 
                                                               replace_stride_with_dilation=[False,False,False])

for p in model.backbone.parameters():
    p.requires_grad = False
model.backbone.eval()
print('model.backbone.eval()')

# +
clip_model, preprocess = clip.load('RN50', device, jit=False)

cur_itrs = 0
# writer = SummaryWriter()

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
if resize_feat:
    t1 = nn.AdaptiveAvgPool2d((56, 56))
else:
    t1 = lambda x: x
# -

model = model.to(device)

for i,(img_id, tar_id, images, labels) in enumerate(train_loader):

    # print(images.shape)
    # torch.Size([32, 3, 768, 768])

    print(i)
    f1 = model.backbone(images.to(device),
                        trunc1=False, trunc2=False, trunc3=False, trunc4=False,
                        get1=True, get2=False, get3=False, get4=False)  # (B,C1,H1,W1)

    # print(f"f1 shape : {f1.shape}")

    #optimize mu and sigma of target features with CLIP
    model_pin_1 = PIN([f1.shape[0],256,1,1], f1.to(device)) #  mu_T (B,C1)  sigma_T(B,C1)
    model_pin_1.to(device)


    optimizer_pin_1 = torch.optim.SGD(params=[
        {'params': model_pin_1.parameters(), 'lr': 1},
    ], lr= 1, momentum=0.9, weight_decay=weight_decay)
        
    if i == len(train_loader)-1 and f1.shape[0] < batch_size :
        style_feature_list = style_feature_list[:f1.shape[0]]

    # print(f"text_source shape : {text_source.shape}")

    while cur_itrs< total_it: 

        cur_itrs += 1

        optimizer_pin_1.zero_grad()

        f1_hal = model_pin_1()
        f1_hal_trans = t1(f1_hal)

        # target_features (optimized)
        target_features_from_f1 = model.backbone(f1_hal_trans,
                                                 trunc1=True, trunc2=False, trunc3=False, trunc4=False,
                                                 get1=False, get2=False, get3=False, get4=False)

        target_features_from_f1 /= target_features_from_f1.norm(dim=-1, keepdim=True).clone().detach()

        # print(f"target_features_from_f1 : {target_features_from_f1.shape}")

        # loss
        loss_CLIP1 = (1- torch.cosine_similarity(style_feature_list, target_features_from_f1, dim=1)).mean()

        loss_CLIP1.backward(retain_graph=True)

        optimizer_pin_1.step()

    cur_itrs = 0

    for name, param in model_pin_1.named_parameters():
        if param.requires_grad and name == 'style_mean':
            learnt_mu_f1 = param.data
        elif param.requires_grad and name == 'style_std':
            learnt_std_f1 = param.data

    for k in range(learnt_mu_f1.shape[0]):
        learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
        learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())

        stats = {}
        stats['mu_f1'] = learnt_mu_f1_
        stats['std_f1'] = learnt_std_f1_

        with open(save_dir+'/'+img_id[k].split('/')[-1]+'.pkl', 'wb') as f:
            pickle.dump(stats, f)
