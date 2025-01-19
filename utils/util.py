import torch
import os
import subprocess
import shutil
import time
from collections import OrderedDict
import json
import torch.optim as optim
import pandas as pd
from model.models import BDenseNet, DenseNet, BEfficientNet, EfficientNet, Model_DA
import csv
import numpy as np


def write_score(writer, iter, mode, metrics):
    writer.add_scalar(mode + '/loss', metrics.data['loss'], iter)
    writer.add_scalar(mode + '/acc', metrics.data['correct'] / metrics.data['total'], iter)


def write_train_val_score(writer, epoch, train_stats, val_stats): #这是K-fold交叉验证的时候用的
    writer.add_scalars('Loss', {'train': train_stats[0],
                                'val': val_stats[0],
                                }, epoch)
    writer.add_scalars('Coeff', {'train': train_stats[1],
                                 'val': val_stats[1],
                                 }, epoch)

    writer.add_scalars('Air', {'train': train_stats[2],
                               'val': val_stats[2],
                               }, epoch)

    writer.add_scalars('CSF', {'train': train_stats[3],
                               'val': val_stats[3],
                               }, epoch)
    writer.add_scalars('GM', {'train': train_stats[4],
                              'val': val_stats[4],
                              }, epoch)
    writer.add_scalars('WM', {'train': train_stats[5],
                              'val': val_stats[5],
                              }, epoch)
    return


def showgradients(model):
    for param in model.parameters():
        print(type(param.data), param.size()) #打印参数的类型和大小
        print("GRADS= \n", param.grad) #打印参数的梯度

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

def save_checkpoint(state, is_best, path,  filename='last'):

    name = os.path.join(path, filename+'_checkpoint.pth.tar')
    print(name)
    torch.save(state, name)

def save_model(model,optimizer, args, metrics, epoch, best_pred_loss,confusion_matrix):
    loss = metrics.data['bacc']
    save_path = args.save
    make_dirs(save_path)
    
    with open(save_path + '/training_arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    is_best = False
    #print(loss)
    #print(best_pred_loss)
    if loss > best_pred_loss:
        is_best = True
        best_pred_loss = loss
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'metrics': metrics.data },
                        is_best, save_path, args.model + "_best")
        np.save(save_path + 'best_confusion_matrix.npy',confusion_matrix.cpu().numpy())
            
    else:
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'metrics': metrics.data},
                        False, save_path, args.model + "_last")

    return best_pred_loss

def load_model(args,n_dbs=1):
    checkpoint = torch.load(args.saved_model)
    model,bflag = select_model(args,n_dbs)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = select_optimizer(args,model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch, bflag

def make_dirs(path):
    if not os.path.exists(path):

        os.makedirs(path)

def create_stats_files(path):
    train_f = open(os.path.join(path, 'train.csv'), 'w')
    val_f = open(os.path.join(path, 'val.csv'), 'w')
    return train_f, val_f #返回的是两个文件

def read_json_file(fname):
    with open(fname, 'r') as handle:
        return json.load(handle, object_hook=OrderedDict) #返回的是一个有序字典

def write_json_file(content, fname):
    with open(fname, 'w') as handle:
        json.dump(content, handle, indent=4, sort_keys=False) #将content写入文件

def read_filepaths(file):
    paths, labels, dbs = [], [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for line in lines:
            if ('/ c o' in line):
                break
            #print(line)    
            #_, path, label= line.split(' ')
            #db = path.split('_')[0]
            _, path, label, db = line.split(' ')
            paths.append(path)
            labels.append(label)
            dbs.append(db)
    labes_array = np.array(labels)
    classes = np.unique(labes_array)
    for i in classes:
        print('Clase={}-Samples={}'.format(i, np.sum(labes_array == i)))
    return paths, labels, dbs #返回的是路径，标签，数据库

def select_model(args,n_dbs=1):
    if args.model == 'BDenseNet':
        if args.init_from:
            model, bflag = BDenseNet(n_classes = args.classes, saved_model = args.saved_model), True
            #输出的是一个贝叶斯模型，bfalg是一个标志位，True表示贝叶斯模型，False表示频率模型
        else:
            model, bflag = BDenseNet(args.classes), True #Flag: True: Bayesian model, False: Frequentist model
    elif args.model == 'DenseNet':
        model, bflag = DenseNet(n_classes = args.classes), False
    elif args.model == 'EfficientNet':
        model, bflag = EfficientNet(n_classes = args.classes), False
    elif args.model == 'BEfficientNet':
        if args.init_from:
            model, bflag = BEfficientNet(n_classes = args.classes, saved_model = args.saved_model), True
        else:
            model, bflag = BEfficientNet(n_classes = args.classes), True

    if args.mode == 'DA':
        return Model_DA(model,n_dbs), bflag
    else:
        return model, bflag

def select_optimizer(args, model):
    if args.opt == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def print_stats(args, epoch, num_samples, trainloader, metrics):
    if (num_samples % args.log_interval == 1):
        print("Epoch:{:2d}\tSample:{:5d}/{:5d}\tLoss:{:.4f}\tAccuracy:{:.2f}".format(epoch,
                                                                                         num_samples,
                                                                                         len(
                                                                                             trainloader) * args.batch_size,
                                                                                         metrics.data[
                                                                                             'loss'] / num_samples,
                                                                                         metrics.data[
                                                                                             'correct'] /
                                                                                         metrics.data[
                                                                                             'total']))
        
def print_summary(args, epoch, num_samples, metrics, mode=''):
    print(mode + "\n SUMMARY EPOCH:{:2d}\tSample:{:5d}/{:5d}\tLoss:{:.4f}\tAccuracy:{:.2f}\tBalancedAccuracy:{:.2f}\n".format(epoch,
                                                                                                     num_samples,
                                                                                                     num_samples ,
                                                                                                     metrics.data[
                                                                                                         'loss'] / num_samples,                                                                             
                                                                                                     metrics.data[
                                                                                                         'correct'] /
                                                                                                     metrics.data[
                                                                                                         'total'],
                                                                                                     metrics.data[
                                                                                                         'bacc']/num_samples))

def ImportantOfContext(ReMap: np.array, Mask: np.array) -> float:
    (rr,cr) = ReMap.shape
    (rm,cm) = Mask.shape
    assert rr == rm, 'Relevance Map and Mask mismatch in the number of rows'
    assert cr == cm, 'Relevance Map and Mask mismatch in the number of columns'
    
    Mask[Mask>0] = 1
    ReMap[ReMap<0]=0 #Take only pixels with positive relevance to estimate IoC
    
    Pin = ReMap * Mask
    npin = np.sum(Pin > 0)
    
    Pout = ReMap * (1 - Mask)
    npout = np.sum(Pout > 0)
    
    IoC = (np.sum(Pout)/npout)/(np.sum(Pin)/npin)
    return IoC

def confusion_matrix(nb_classes):#nb_classes是类别数

    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    print(confusion_matrix)

def BalancedAccuray(CM):
    Nc = CM.shape[0]
    BACC = np.zeros(Nc)
    for i in range(Nc):
        BACC[i] = CM[i,i]/np.sum(CM[i,:])
    print(np.mean(BACC))
    return np.mean(BACC)

class Metrics:
    def __init__(self, path, keys=None, writer=None):
        self.writer = writer

        self.data = {'correct': 0,
                     'total': 0,
                     'loss': 0,
                     'accuracy': 0,
                     'bacc':0,
                     }
        self.save_path = path

    def reset(self):
        for key in self.data:
            self.data[key] = 0

    def update_key(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self.data[key] += value

    def update(self, values):
        for key in self.data:
            self.data[key] += values[key]
    
    def replace(self, values):
        for key in values:
            self.data[key] = values[key]  

    def avg_acc(self):
        return self.data['correct'] / self.data['total']

    def avg_loss(self):
        return self.data['loss'] / self.data['total']

    def save(self):
        with open(self.save_path, 'w') as save_file:
            a = 0  # csv.writer()
            # TODO

def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2, wait=False, sleep_time=10):
    """
    Adjusted to work in environments without GPUs.
    """
    import subprocess
    import os
    import time

    def _check():
        try:
            # Try to get GPU info using nvidia-smi
            smi_query_result = subprocess.check_output(
                "nvidia-smi -q -d Memory", shell=True
            )
            gpu_info = smi_query_result.decode("utf-8").split("\n")
            gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
            gpu_info = [
                int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
            ]  # Parse memory usage
            free_gpus = [
                str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
            ]
            return free_gpus[: min(max_gpus, len(free_gpus))]
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If nvidia-smi fails, assume no GPU is available
            print("No GPU detected or nvidia-smi not available. Using CPU.")
            return []

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        # No GPU is available, disable GPU usage
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("No free GPUs found. Using CPU only.")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus_to_use)
        print(f"Using GPU(s): {','.join(gpus_to_use)}")

"""
def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2, wait=False, sleep_time=10):
    #这段代码是为了找到空闲的GPU，然后将其分配给当前进程
    
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere

    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        free_gpus = [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ]
        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join(free_gpus)
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    print(f"Using GPU(s): {gpus_to_use}")
    #logger.info(f"Using GPU(s): {gpus_to_use}")
"""