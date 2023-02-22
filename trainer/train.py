import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from model.loss import crossentropy_loss
from utils.util import Metrics, print_stats, print_summary, select_model, select_optimizer, load_model, assign_free_gpus
from model.metric import accuracy
from COVIDXDataset.dataset import COVIDxDataset
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
from bayesian_torch.models.dnn_to_bnn import get_kl_loss


def initialize(args):
    if args.device is not None:
        assign_free_gpus()
    model, bflag = select_model(args)
    
    optimizer = select_optimizer(args,model)
    if (args.cuda):
        model.cuda()
   
    train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=args.dataset,
                                 dim=(224, 224), pre_processing = args.pre_processing)
    #print(train_loader.)
    #------ Class weigths for sampling and for loss function -----------------------------------
    labels = np.unique(train_loader.labels)
    print(labels)
    class_weight = compute_class_weight(class_weight='balanced', classes=labels, y=train_loader.labels)
    #weights_sample = compute_sample_weight('balanced',train_loader.labels)
    #print(np.unique(weights_sample))
    #---------- Alphabetical order in labels does not correspond to class order in COVIDxDataset-----
    class_weight = class_weight[::-1]
    #---------------------------------------------------------------------------------
    #weights_sample = torch.DoubleTensor(weights_sample)                                       
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_sample, len(weights_sample))                       
    if (args.cuda):
        class_weight = torch.from_numpy(class_weight.astype(float)).cuda()
    else:
        class_weight = torch.from_numpy(class_weight.astype(float))
    #print(class_weight.shape)
    #-------------------------------------------
    val_loader = COVIDxDataset(mode='test', n_classes=args.classes, dataset_path=args.dataset,
                               dim=(224, 224), pre_processing = args.pre_processing)
    #------------------------------------------------------------------------------------
    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 4}#'sampler' : sampler
    
    test_params = {'batch_size': args.batch_size,
                   'shuffle': True,
                   'num_workers': 4}
    #------------------------------------------------------------------------------------------
    training_generator = DataLoader(train_loader, **train_params)
    val_generator = DataLoader(val_loader, **test_params)
    return model, optimizer,training_generator,val_generator, class_weight, bflag

def train(args, model, trainloader, optimizer, epoch, class_weight):
    model.train()
    #criterion = nn.CrossEntropyLoss(weight=class_weight,reduction='mean')

    metrics = Metrics('')
    metrics.reset()
    #-------------------------------------------------------
    #Esto es para congelar las capas de la red preentrenada
    #for m in model.modules():
    #    if isinstance(m, nn.BatchNorm2d):
    #        m.train()
    #        m.weight.requires_grad = False
    #        m.bias.requires_grad = False
    #-----------------------------------------------------



    for batch_idx, input_tensors in enumerate(trainloader):
        optimizer.zero_grad()
        input_data, target = input_tensors
        if (args.cuda):
            input_data = input_data.cuda()
            target = target.cuda()
        #print(input_data.shape)
        output = model(input_data)
        #print(output.shape)
        #print(target.shape)
        
        loss = crossentropy_loss(output, target,weight=class_weight)
        loss.backward()
        optimizer.step()
        correct, total, acc = accuracy(output, target)

        num_samples = batch_idx * args.batch_size + 1
        _, output_class = output.max(1)
        #print(output_class)
        #print(target)
        bacc = balanced_accuracy_score(target.cpu().detach().numpy(),output_class.cpu().detach().numpy())
        metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})
        print_stats(args, epoch, num_samples, trainloader, metrics)

    print_summary(args, epoch, num_samples, metrics, mode="Training")
    return metrics

def train_bayesian(args, model, trainloader, optimizer, epoch, class_weight):
    model.train()
    #criterion = nn.CrossEntropyLoss(weight=class_weight,reduction='mean')

    metrics = Metrics('')
    metrics.reset()
    
    for batch_idx, input_tensors in enumerate(trainloader):
        optimizer.zero_grad()
        input_data, target = input_tensors
        if (args.cuda):
            input_data = input_data.cuda()
            target = target.cuda()
        #print(input_data.shape)
        output = model(input_data)
        
        kl = get_kl_loss(model)
        ce_loss = crossentropy_loss(output, target,weight=class_weight)
        loss = ce_loss + kl / args.batch_size 
        
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output_mc = []
            for _ in range(args.n_monte_carlo):
                logits = model(input_data)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                output_mc.append(probs)
            output = torch.stack(output_mc)  
            pred_mean = output.mean(dim=0)

        correct, total, acc = accuracy(pred_mean, target)

        num_samples = batch_idx * args.batch_size + 1
        _, output_class = pred_mean.max(1)
        #print(output_class)
        #print(target)
        bacc = balanced_accuracy_score(target.cpu().detach().numpy(),output_class.cpu().detach().numpy())
        metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})
        print_stats(args, epoch, num_samples, trainloader, metrics)

    print_summary(args, epoch, num_samples, metrics, mode="Training")
    return metrics

def validation(args, model, testloader, epoch, class_weight):
    model.eval()

    #-------------------------------------------------------
    #Esto es para congelar las capas de la red preentrenada
    #for m in model.modules():
    #    if isinstance(m, nn.BatchNorm2d):
    #        m.train()
    #        m.weight.requires_grad = False
    #        m.bias.requires_grad = False
    #-----------------------------------------------------

    criterion = nn.CrossEntropyLoss(size_average='mean')

    metrics = Metrics('')
    metrics.reset()
    confusion_matrix = torch.zeros(args.classes, args.classes)
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            input_data, target = input_tensors
            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda()
            #print(input_data.shape)
            output = model(input_data)
            
            loss = crossentropy_loss(output, target,weight=class_weight)

            correct, total, acc = accuracy(output, target)
            #num_samples = batch_idx * args.batch_size + 1
            _, preds = torch.max(output, 1)
            bacc = balanced_accuracy_score(target.cpu().detach().numpy(),preds.cpu().detach().numpy())
            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})
            #print_stats(args, epoch, num_samples, testloader, metrics)

    print_summary(args, epoch, batch_idx, metrics, mode="Validation")
    return metrics,confusion_matrix

def validation_bayesian(args, model, testloader, epoch, class_weight):
    model.eval()

    metrics = Metrics('')
    metrics.reset()
    confusion_matrix = torch.zeros(args.classes, args.classes)
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            input_data, target = input_tensors
            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda()
            #print(input_data.shape)
            output = model(input_data)
            loss = crossentropy_loss(output, target,weight=class_weight)

            with torch.no_grad():
                output_mc = []
                for _ in range(args.n_monte_carlo):
                    logits = model(input_data)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    output_mc.append(probs)
                output = torch.stack(output_mc)  
                pred_mean = output.mean(dim=0)

            correct, total, acc = accuracy(pred_mean, target)
            num_samples = batch_idx * args.batch_size + 1
            _, preds = torch.max(pred_mean, 1)
            bacc = balanced_accuracy_score(target.cpu().detach().numpy(),preds.cpu().detach().numpy())
            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})
            #print_stats(args, epoch, num_samples, testloader, metrics)

    print_summary(args, epoch, num_samples, metrics, mode="Validation")
    return metrics,confusion_matrix
    
def initialize_from_saved_model(args):
    print('Training on saved model')
    if args.device is not None:
        assign_free_gpus()
    model, optimizer, epoch, bflag = load_model(args)
       
    train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=args.dataset,
                                 dim=(224, 224), pre_processing = args.pre_processing)
    #print(train_loader.)
    #------ Class weigths for sampling and for loss function -----------------------------------
    labels = np.unique(train_loader.labels)
    #print(labels)
    class_weight = compute_class_weight(class_weight='balanced', classes=labels, y=train_loader.labels)
    class_weight = class_weight[::-1]
    #class_weight[2]=50
    #weights = torch.DoubleTensor(class_weight.copy())
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_loader.labels))
    if (args.cuda):
        class_weight = torch.from_numpy(class_weight.astype(float)).cuda()
    else:
        class_weight = torch.from_numpy(class_weight.astype(float))
    #print(class_weight.shape)
    #-------------------------------------------
    val_loader = COVIDxDataset(mode='test', n_classes=args.classes, dataset_path=args.dataset,
                            dim=(224, 224), pre_processing = args.pre_processing)
    #------------------------------------------------------------------------------------
    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 4}#'sampler' : sampler
    test_params = {'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': 4}
    #------------------------------------------------------------------------------------------
    training_generator = DataLoader(train_loader, **train_params)
    val_generator = DataLoader(val_loader, **test_params)
    return model, optimizer,training_generator,val_generator, class_weight, epoch, bflag