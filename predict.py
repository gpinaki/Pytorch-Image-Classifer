#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER: Pinaki Guha
# DATE CREATED:  April 14, 2020                                
# REVISED DATE: 
# PURPOSE: Building the command line application. 
# predict.py, will predict a new network on a dataset and save the model as a checkpoint.

"""
@author: Pinaki Guha
@title: Image Classifier training file
"""
# ------------------------------------------------------------------------------- #
# Import Libraries
# ------------------------------------------------------------------------------- #

import argparse
import json
import PIL
import torch
import numpy as np
import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from math import ceil
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def arg_parser():
    
    parser = argparse.ArgumentParser(description='Image Classifier Predict Module')

    parser.add_argument('--image', 
                        type=str, 
                        action='store',
                        help='Impage file path for prediction.',
                        required=True)

    parser.add_argument('--checkpoint', 
                        type=str, 
                        action='store',
                        default="checkpoint_saved.pth",
                        help='Checkpoint file name as str.')
    
    parser.add_argument('--top_k', 
                        type=int, 
                        action='store',
                        default=3,
                        help='Choose top K matches as int.')
    
    parser.add_argument('--category_names', 
                        type=str, 
                        action='store',
                        default="cat_to_name.json",
                        help='Mapping from categories to real flower names.')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        default=True,
                        help='Use GPU for Calculations')

    # Parse args
    args = parser.parse_args()
    
    return args

def load_checkpoint(filepath):
    
    checkpoint = torch.load('checkpoint_saved.pth')
    
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
        
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    print("Start processing image")
    
    
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    print("before proprocess")
    image = preprocess(image)
    print("after proprocess")
    return image

def predict(image, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''  
    
    # Convert 2D image to 1D vector
    img = np.expand_dims(image, 0)
    img = torch.from_numpy(img)
    model.eval()
    model = model.to(device)
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    
    probs = topk[0][0].tolist()
    classes = topk[1][0].tolist()
    
    return probs,classes

def main():
    print("Main program started")
    
     ## Grab input parameters from parser
    args = arg_parser()
    print(args)
    
    # Get Flower Names
    cat_to_name = args.category_names
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)
        
    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)
    
    img = PIL.Image.open(args.image)
    
    # Process Image
    img = process_image(img)
    
    train_on_gpu = args.gpu
    device = torch.device("cuda" if train_on_gpu else "cpu")
    print("Device is: "+ str(device)+ ". Train on GPU is "+ str(train_on_gpu))
    
    # Print out probabilities
    top_k_int = args.top_k
    
    probs, classes = predict(img, model,device, top_k_int)
    
    print(probs)
    print(classes)
    top_flower = []
    for c in classes:
        top_flower.append(cat_to_name[str(c)])
    
    print(top_flower)
        
        
if __name__ == "__main__":
    main()

    