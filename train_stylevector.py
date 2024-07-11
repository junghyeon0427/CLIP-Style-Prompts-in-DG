# -*- coding: utf-8 -*-
import clip
from clip.simple_tokenizer import *

# +
# import custom CLIP
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import os
import pickle
import io


# -

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
device = 'cuda:3'
clip_model, preprocess = clip.load("RN50", device=device)

text_encoder = TextEncoder(clip_model)
tokenizer = SimpleTokenizer()
# -

# training iteration -> argument
L = 100
# number of style vectors -> argument
K = 80
# dimension
D = 512


def style_diversity_loss(style_vectors, new_style_word_vector, i):
    Lstyle = 0.0
    
    si = new_style_word_vector
    si = F.normalize(si, dim=1)
    
    for j in range(i):
        sj = style_vectors[j]
        sj = F.normalize(sj, dim=1)

        Lstyle = Lstyle + abs(si @ sj.T)
    
    Lstyle = Lstyle / i
    
    return Lstyle


from tqdm import tqdm

# +
style_feature_list =[]

for j in tqdm(range(K)):
    ctx_vectors = torch.empty(1, 1, D)
    nn.init.normal_(ctx_vectors, std=0.02)
    # ctx_vectors = ctx_vectors.repeat(7, 1, 1)
    prompt_prefix = " ".join(["X"] * n_ctx)
    
    # optimize vector
    ctx = nn.Parameter(ctx_vectors)
    
    # optimizer & scheduler 
    optimizer = torch.optim.SGD([ctx], lr=0.002, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        
    ctx_half1 = 'Driving conditions at'
    
    # classnames = [name.replace("_", " ") for name in class_name]
    
    for x in range(L):
        
        if j == 0:
            break
        
        prompts = [ctx_half1 + " " + prompt_prefix]
        tokenized_prompts = clip.tokenize(prompts).to(device)
        embedding = clip_model.token_embedding(tokenized_prompts)

        prefix = embedding[:, :4, :].to(device) 
        suffix = embedding[:, 4+n_ctx :, :].to(device)
        
        # "Driving conditions at X"라는 임베딩 값을 "Driving conditions at S*"로 변경
        ctx_i = ctx[:, :, :].to(device)
        prefix_i = prefix[:, :, :].to(device)
        suffix_i = suffix[:, :, :].to(device)
        
        prompts = torch.cat((prefix_i, ctx_i, suffix_i), dim=1).half()

        # style_feature : "Driving conditions at S*"
        style_feature = text_encoder(prompts.half(), tokenized_prompts.long())
        
        # Lstyle loss
        if j != 0:
            Lstyle = style_diversity_loss(style_feature_list, style_feature, j)
        
        optimizer.zero_grad()
        Lstyle.backward()
        optimizer.step()
    
    style_feature_list.append(style_feature.detach())
# -

# pkl 저장
with open('style_feature.pkl', 'wb') as f:
    pickle.dump(style_feature_list, f)
