#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER: Pinaki Guha
# DATE CREATED:  April 14, 2020                                
# REVISED DATE: 
# PURPOSE: Building the command line application. 
# train.py, will train a new network on a dataset and save the model as a checkpoint.

"""
@author: Pinaki Guha
@title: Image Classifier training file
"""
# ------------------------------------------------------------------------------- #
# Import Libraries
# ------------------------------------------------------------------------------- #

import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import torch.nn.functional as F
import PIL

#   Example call:
#   python train.py data_directory
#   Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#   Choose architecture: python train.py data_dir --arch "vgg13"
#   Set hyperparameters:  python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#   se GPU for training:  python train.py data_dir --gpu

def arg_parser():
    
    parser = argparse.ArgumentParser(description='Image Classifier Training Module')
    
    parser.add_argument('--save_dir', 
                        type=str, 
                        action='store',
                        default="cpt",
                        help='Set directory to store checkpoints'
                        )
        
    parser.add_argument('--arch', 
                        type=str, 
                        action='store',
                        default="vgg",
                        help='Choose network architecture as str'
                        )
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        action='store',
                        default = 0.0001,
                        help='Give learning rate as float')
    
    parser.add_argument('--hidden_units', 
                        type=int, 
                        action='store',
                        default = 512,
                        help='Hidden units for classifier as int')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        action='store',
                        default = 2,
                        help='Number of epochs for training as int')
    
    parser.add_argument('--gpu', 
                        action="store_true", 
                        default = True,
                        help='Use GPU + Cuda for calculations as Boolean')
    
    args = parser.parse_args()
    return args

def train_transform(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 
    
    train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
    return train_datasets
    
def test_transform(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                    ])
    test_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)
    return test_datasets


def dataloader(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,shuffle=True)
    return dataloader
    
def model_loader(architecture):
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.required_grad = False
            
        print("Network architecture specified as vgg16.")
        return model
    
def build_classifier(model,hidden_units):

    if type(hidden_units) == type(None): 
        hidden_units = 4096 
    print("Number of Hidden Layers :"+str(hidden_units))
    # Find Input Layers
    input_features = model.classifier[0].in_features
   
    print("Input features: "+ str(input_features))
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier
    

def validate_model(model, validloader, device):
    
    criterion = nn.NLLLoss()
    model.eval()
    accuracy = 0 
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        logps = model(images)
        loss = criterion(logps,labels)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()

    print("Test accuracy: {:.3f}".format(accuracy/len(validloader)))
    return None
    
def save_model(model,train_dataset, save_dir):
    model.class_to_idx = train_dataset.class_to_idx
    
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                   'epochs': 1,
                   'batch_size': 64,
                   'model': models.vgg16(pretrained=True),
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
   
    torch.save(checkpoint, 'checkpoint_saved.pth')
    print("Model is saved")
    
def main():
    print("Main program started")
    
    ## Grab input parameters from parser
    args = arg_parser()
    print(args)
    
    ## Initialize directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Get training, test and validation dataset
    train_dataset = train_transform(train_dir)
    test_dataset = test_transform(test_dir)
    valid_dataset = test_transform(valid_dir)
    
    print("train dataset size: "+ str(len(train_dataset)))
    print("test dataset size: "+ str(len(test_dataset)))
    print("valid dataset size: "+ str(len(valid_dataset)))
    
    #Now load trainloader, testloader and validation loaders
    trainloader = dataloader(train_dataset)
    testloader = dataloader(test_dataset)
    validloader = dataloader(valid_dataset)
    
    #images, labels = next(iter(testloader))
    #print(images.size())
    
    # Create the model
    model = model_loader(args.arch)
    
    for param in model.parameters():
        param.requires_grad = False 

    # Initialize the classifier
    model.classifier = build_classifier(model,args.hidden_units)
    
    train_on_gpu = args.gpu
    device = torch.device("cuda" if train_on_gpu else "cpu")
    print("Device is: "+ str(device)+ ". Train on GPU is "+ str(train_on_gpu))
    
    # Define loss
    criterion = nn.NLLLoss()
    
    # Define Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    model.to(device)
    
    epochs = args.epochs
    steps = 0
    running_loss = 0.0
    # print after how many steps
    print_every = 25
    training_losses, test_losses = [], []
    
    for epoch in range(epochs):
    
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            model.train()
            ## Reset gradiants to zero before every step
            optimizer.zero_grad()
            
            logps = model(images)
            
            loss = criterion(logps,labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if steps % print_every == 0:
            
                # Turn model into evaluation
                model.eval()
                test_loss = 0
                accuracy = 0
                for images , labels in testloader:
                
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    loss = criterion(logps,labels)
                    test_loss += loss.item()
                
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()
                    
                    training_losses.append(running_loss)
                    test_losses.append(test_loss)
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")              
                running_loss = 0
                
                model.train()
                
    print("\nTraining model is now completed.")   
    
    validate_model(model, validloader, device)
    save_model(model,train_dataset,args.save_dir)
    
    
if __name__ == "__main__":
    main()
