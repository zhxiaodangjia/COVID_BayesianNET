import torch
import os
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from model.loss import crossentropy_loss
from utils.util import Metrics, print_stats, print_summary, select_model, select_optimizer, load_model, assign_free_gpus
from model.metric import accuracy
from COVIDXDataset.dataset import COVIDxDataset, COVIDxDataset_DA
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
from bayesian_torch.models.dnn_to_bnn import get_kl_loss


def initialize(args):
    if args.device is not None:
            assign_free_gpus()

    if args.mode == 'DA':

        train_loader = COVIDxDataset_DA(mode='train', n_classes=args.classes, dataset_path=args.dataset,
                                 dim=(224, 224), pre_processing = args.pre_processing)        
        labels_db = np.unique(train_loader.dbs)
        db_weight = compute_class_weight(class_weight='balanced', classes=labels_db, y=train_loader.dbs)
        n_dbs = len(labels_db)
        print(f'Numer of different databases = {n_dbs}')
        if args.resume:
            model, optimizer, epoch, bflag = load_model(args,n_dbs)
        else:
            model, bflag = select_model(args,n_dbs)
            optimizer = select_optimizer(args,model)
            epoch = 0
            
        #---------------------------------------------------------------------------------                    
        if (args.cuda):
            db_weight = torch.from_numpy(db_weight.astype(float)).cuda()
        else:
            db_weight = torch.from_numpy(db_weight.astype(float))
    else:
        if args.resume:
            model, optimizer, epoch, bflag = load_model(args)
        else:
            model, bflag = select_model(args)
            optimizer = select_optimizer(args,model)
            epoch = 0
    
        train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=args.dataset,
                                    dim=(224, 224), pre_processing = args.pre_processing)
                    
    #------ Class weigths for sampling and for loss function -----------------------------------
    labels = np.unique(train_loader.labels)
    class_weight = compute_class_weight(class_weight='balanced', classes=labels, y=train_loader.labels)
    #---------- Alphabetical order in labels does not correspond to class order in COVIDxDataset-----
    class_weight = class_weight[::-1]
    #---------------------------------------------------------------------------------                    
    if (args.cuda):
        model.cuda()
        class_weight = torch.from_numpy(class_weight.astype(float)).cuda()
    else:
        class_weight = torch.from_numpy(class_weight.astype(float))
    
    if args.mode == 'DA':
        weights = [class_weight,db_weight]
    else:
        weights = class_weight
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

    return model, optimizer,training_generator,val_generator, weights, epoch, bflag
   
    

def train(args, model, trainloader, optimizer, epoch, weights):
    model.train()
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

    if args.mode == 'DA':
        for batch_idx, input_tensors in enumerate(trainloader):
            optimizer.zero_grad()
            input_data, target = input_tensors
            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda() 

            output_class, output_db = model(input_data)
     
            loss_class = crossentropy_loss(output_class, target[:,0], weight=weights[0])
            loss_db = crossentropy_loss(output_db,  target[:,1], weight=weights[1])
            loss = loss_class + loss_db
            loss.backward()
            optimizer.step()
            correct, total, acc = accuracy(output_class, target[:,0])

            num_samples = batch_idx * args.batch_size + 1
            _, predicted_class = output_class.max(1)

            bacc = balanced_accuracy_score(target[:,0].cpu().detach().numpy(),predicted_class.cpu().detach().numpy())
            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})
            print_stats(args, epoch, num_samples, trainloader, metrics)
    else:
        for batch_idx, input_tensors in enumerate(trainloader):
            optimizer.zero_grad()
            input_data, target = input_tensors
            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda()

            output = model(input_data)
            loss = crossentropy_loss(output, target,weight=weights)
            loss.backward()
            optimizer.step()
            correct, total, acc = accuracy(output, target)

            num_samples = batch_idx * args.batch_size + 1
            _, predicted_class = output.max(1)

            bacc = balanced_accuracy_score(target.cpu().detach().numpy(),predicted_class.cpu().detach().numpy())
            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})
            print_stats(args, epoch, num_samples, trainloader, metrics)

    print_summary(args, epoch, num_samples, metrics, mode="Training")
    return metrics

def train_bayesian(args, model, trainloader, optimizer, epoch, weights):
    model.train()
    metrics = Metrics('')
    metrics.reset()
    
    if args.mode == 'DA':
        for batch_idx, input_tensors in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            input_data, target = input_tensors
            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda()
  
            output_class, output_db = model(input_data)
            kl = get_kl_loss(model)
            ce_loss = crossentropy_loss(output_class, target[:,0], weight=weights[0])
            loss_db = crossentropy_loss(output_db, target[:,1], weight=weights[1])
            loss = loss_db + ce_loss + kl / args.batch_size 
            
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

            correct, total, acc = accuracy(pred_mean, target[:,0])

            num_samples = batch_idx * args.batch_size + 1
            _, predicted_class = pred_mean.max(1)

            bacc = balanced_accuracy_score(target[:,0].cpu().detach().numpy(),predicted_class.cpu().detach().numpy())
            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})
            print_stats(args, epoch, num_samples, trainloader, metrics)
    else:
        for batch_idx, input_tensors in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            input_data, target = input_tensors
            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda()
            
            output = model(input_data)
            kl = get_kl_loss(model)
            ce_loss = crossentropy_loss(output, target,weight=weights)
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
            _, predicted_class = pred_mean.max(1)
            bacc = balanced_accuracy_score(target.cpu().detach().numpy(),predicted_class.cpu().detach().numpy())
            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})
            print_stats(args, epoch, num_samples, trainloader, metrics)

    print_summary(args, epoch, num_samples, metrics, mode="Training")
    return metrics

def validation(args, model, testloader, epoch, class_weight):
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

def validation_bayesian(args, model, testloader, epoch, weights):
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
            loss = crossentropy_loss(output, target,weight=weights[0])

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
    
