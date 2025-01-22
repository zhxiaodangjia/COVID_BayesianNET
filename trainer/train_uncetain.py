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
from utils.avuc_loss import predictive_entropy, mutual_information

##所有的db都是指的数据集来源，也就是数据集的标签
def initialize(args):
    if args.device is not None:
            assign_free_gpus()

    if args.mode == 'DA':

        train_loader = COVIDxDataset_DA(mode='train', n_classes=args.classes, dataset_path=args.dataset,
                                 dim=(224, 224), pre_processing = args.pre_processing)        
        labels_db = np.unique(train_loader.dbs) #数据集来源的标签
        db_weight = compute_class_weight(class_weight='balanced', classes=labels_db, y=train_loader.dbs)
        #计算数据集来源的权重
        n_dbs = len(labels_db) #number of different databases
        print(f'Numer of different databases = {n_dbs}')
        if args.resume: #如果是继续训练
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

            output_class, output_db = model(input_data) #output_db是数据集来源的预测
     
            loss_class = crossentropy_loss(output_class, target[:,0], weight=weights[0]) #target的shape是(batch_size,2),第一个是类别，第二个是数据集来源
            loss_db = crossentropy_loss(output_db,  target[:,1], weight=weights[1])#target的shape是(batch_size,2)
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
    # 设置模型为训练模式（启用 dropout 等训练相关模块）
    model.train()

    # 初始化评估指标对象
    metrics = Metrics('')
    metrics.reset()

    # 如果使用领域自适应（DA）模式
    if args.mode == 'DA':
        # 遍历训练数据加载器中的每个批次
        for batch_idx, input_tensors in enumerate(trainloader):
            # 确保模型处于训练模式
            model.train()
            
            # 清空优化器的梯度
            optimizer.zero_grad()

            # 从输入张量中解包数据和目标
            input_data, target = input_tensors
            
            # 如果使用 GPU，将数据和目标迁移到 CUDA 设备
            if args.cuda:
                input_data = input_data.cuda()
                target = target.cuda()

            # 模型前向传播，生成类别输出和领域判别器输出
            output_class, output_db = model(input_data)

            # 计算 KL 散度正则化损失（控制分布的平滑性）
            kl = get_kl_loss(model)

            # 计算分类交叉熵损失
            ce_loss = crossentropy_loss(output_class, target[:, 0], weight=weights[0])
            
            # 计算领域分类交叉熵损失
            loss_db = crossentropy_loss(output_db, target[:, 1], weight=weights[1])
            
            # 总损失为分类损失、领域损失和 KL 损失之和
            loss = loss_db + ce_loss + kl / args.batch_size

            # 反向传播计算梯度
            loss.backward()

            # 更新模型参数
            optimizer.step()

            # 设置模型为评估模式（禁用 dropout 等）
            model.eval()

            # 使用 Monte Carlo 采样估计不确定性
            with torch.no_grad():
                output_mc = []
                for _ in range(args.n_monte_carlo):
                    # 对输入数据进行前向传播，生成 logits
                    logits = model(input_data)

                    # 使用 softmax 将 logits 转换为概率分布
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                    # 将结果追加到 MC 输出列表中
                    output_mc.append(probs)

                # 将 MC 采样结果堆叠为张量，形状为 [T, B, C] （采样次数，批大小，类别数）
                output = torch.stack(output_mc)

                # 计算 MC 采样的均值，形状为 [B, C]（最终预测分布）
                pred_mean = output.mean(dim=0)

            # 计算预测的准确样本数、总样本数和准确率
            correct, total, acc = accuracy(pred_mean, target[:, 0])

            # 当前批次处理的样本总数
            num_samples = batch_idx * args.batch_size + 1

            # 获取预测类别（取概率最大的类别作为预测值）
            _, predicted_class = pred_mean.max(1)

            # 计算加权平均精度（balanced accuracy score）
            bacc = balanced_accuracy_score(target[:, 0].cpu().detach().numpy(),
                                           predicted_class.cpu().detach().numpy())

            # 更新评估指标，包括正确样本数、总样本数、损失、准确率等
            metrics.update({
                'correct': correct,
                'total': total,
                'loss': loss.item(),
                'accuracy': acc,
                'bacc': bacc
            })

            # 打印当前批次的训练统计信息
            print_stats(args, epoch, num_samples, trainloader, metrics)

    else:
        # 如果不使用领域自适应（DA）模式
        for batch_idx, input_tensors in enumerate(trainloader):
            # 确保模型处于训练模式
            model.train()

            # 清空优化器的梯度
            optimizer.zero_grad()

            # 从输入张量中解包数据和目标
            input_data, target = input_tensors
            
            # 如果使用 GPU，将数据和目标迁移到 CUDA 设备
            if args.cuda:
                input_data = input_data.cuda()
                target = target.cuda()

            # 模型前向传播，生成输出 logits！！如果使用dnn_to_bnn，这里的output和KL就要分开写！！！
            output = model(input_data)

            # 计算 KL 散度正则化损失
            kl = get_kl_loss(model)

            # 计算交叉熵损失
            ce_loss = crossentropy_loss(output, target, weight=weights)

            # 总损失为分类损失和 KL 散度损失之和
            loss = ce_loss + kl / args.batch_size

            # 反向传播计算梯度
            loss.backward()

            # 更新模型参数
            optimizer.step()

            # 设置模型为评估模式
            model.eval()

            # 使用 Monte Carlo 采样估计不确定性
            with torch.no_grad():
                output_mc = []
                for _ in range(args.n_monte_carlo):
                    # 对输入数据进行前向传播，生成 logits
                    logits = model(input_data)

                    # 使用 softmax 将 logits 转换为概率分布
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                    # 将结果追加到 MC 输出列表中
                    output_mc.append(probs)

                # 将 MC 采样结果堆叠为张量
                output = torch.stack(output_mc)

                # 计算 MC 采样的均值
                pred_mean = output.mean(dim=0)

            # 计算预测的准确样本数、总样本数和准确率
            correct, total, acc = accuracy(pred_mean, target)

            # 当前批次处理的样本总数
            num_samples = batch_idx * args.batch_size + 1

            # 获取预测类别
            _, predicted_class = pred_mean.max(1)

            # 计算加权平均精度
            bacc = balanced_accuracy_score(target.cpu().detach().numpy(),
                                           predicted_class.cpu().detach().numpy())

            # 更新评估指标
            metrics.update({
                'correct': correct,
                'total': total,
                'loss': loss.item(),
                'accuracy': acc,
                'bacc': bacc
            })

            # 打印当前批次的训练统计信息
            print_stats(args, epoch, num_samples, trainloader, metrics)

    # 打印整个 epoch 的训练统计信息
    print_summary(args, epoch, num_samples, metrics, mode="Training")

    # 返回最终评估指标
    return metrics


def validation(args, model, testloader, epoch, weights):
    model.eval()
    metrics = Metrics('')
    metrics.reset()
    confusion_matrix = torch.zeros(args.classes, args.classes)
    if args.mode == 'DA':
        class_weights = weights[0]
    else:
        class_weights = weights
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            input_data, target = input_tensors
            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda()
            #print(input_data.shape)
            output = model(input_data)
            
            loss = crossentropy_loss(output, target,weight=class_weights)

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
'''
def validation_bayesian(args, model, testloader, epoch, weights):
    model.eval()
    metrics = Metrics('')
    metrics.reset()
    confusion_matrix = torch.zeros(args.classes, args.classes)
    if args.mode == 'DA':
        class_weights = weights[0]
    else:
        class_weights = weights
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            input_data, target = input_tensors
            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda()
            #print(input_data.shape)
            output = model(input_data)
            loss = crossentropy_loss(output, target,weight=class_weights)

            with torch.no_grad():
                output_mc = []
                for _ in range(args.n_monte_carlo):
                    logits = model(input_data) #输出的是logits包含了所有的mc runs
                    probs = torch.nn.functional.softmax(logits, dim=-1) #将logits转换为概率，dim=-1表示对最后一个维度进行softmax
                    output_mc.append(probs)
                output = torch.stack(output_mc)  
                pred_mean = output.mean(dim=0)

            correct, total, acc = accuracy(pred_mean, target)
            num_samples = batch_idx * args.batch_size + 1
            _, preds = torch.max(pred_mean, 1) #取概率最大的类别
            bacc = balanced_accuracy_score(target.cpu().detach().numpy(),preds.cpu().detach().numpy())
            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc, 'bacc':bacc})
            #print_stats(args, epoch, num_samples, testloader, metrics)

    print_summary(args, epoch, num_samples, metrics, mode="Validation")
    return metrics,confusion_matrix
   
'''
def validation_bayesian(args, model, testloader, epoch, weights):
    # 导入不确定性量化工具函数
    

    # 将模型设置为评估模式（禁用 dropout 等影响推理的层）
    model.eval()

    # 初始化评估指标对象
    metrics = Metrics('')
    metrics.reset()

    # 初始化混淆矩阵，用于记录每个类别的预测情况
    confusion_matrix = torch.zeros(args.classes, args.classes)

    # 根据模式选择权重（适用于不同损失函数情况）
    if args.mode == 'DA':
        class_weights = weights[0]  # 使用第一组权重
    else:
        class_weights = weights  # 默认权重

    # 创建字典用于存储不确定性指标
    uncertainties = {"predictive_entropy": [], "mutual_information": []}

    # 禁用梯度计算（加速推理，节省内存）
    with torch.no_grad():
        # 遍历验证集数据加载器
        for batch_idx, input_tensors in enumerate(testloader):
            # 解包输入数据（input_data 是图像，target 是标签）
            input_data, target = input_tensors

            # 如果使用 GPU，将数据迁移到 CUDA 设备
            if args.cuda:
                input_data = input_data.cuda()
                target = target.cuda()

            # 
            # 执行前向传播，计算模型输出 logits, 形状为 [B, C]（批大小，类别数）！！这里有点问题，原模型输出应该是output和kl，这里只有output
            output = model(input_data)

            # 根据交叉熵损失函数计算损失
            loss = crossentropy_loss(output, target, weight=class_weights)

            # 用于存储每次 MC 采样的输出概率分布
            output_mc = []

            # 多次 MC 采样，计算预测分布
            for _ in range(args.n_monte_carlo):
                # 对输入数据进行前向传播
                logits = model(input_data)

                # 使用 softmax 将 logits 转换为概率分布
                probs = torch.nn.functional.softmax(logits, dim=-1)

                # 将结果追加到 MC 输出列表
                output_mc.append(probs)

            # 将 MC 采样结果堆叠为张量，形状为 [T, B, C] （采样次数，批大小，类别数）
            output = torch.stack(output_mc)

            # 计算 MC 采样的均值（即最终预测分布），形状为 [B, C]
            pred_mean = output.mean(dim=0)

            #这里得再看看！！
            #  将 MC 输出的维度调整为 [B, T, C]，方便不确定性计算,这里有待商榷？？因为得看预测熵和互信息取的是哪一唯的值，改变了计算定义 这里就不需要了
            output_mc_np = output.permute(1, 0, 2).cpu().numpy()

            # 批量计算每个样本的不确定性指标
            batch_predictive_unc = []  # 用于存储预测熵
            batch_model_unc = []  # 用于存储模型不确定性
            for sample_mc_preds in output_mc_np:
                # 对每个样本计算预测熵
                batch_predictive_unc.append(predictive_entropy(sample_mc_preds))

                # 对每个样本计算互信息
                batch_model_unc.append(mutual_information(sample_mc_preds))

            # 将当前批次的不确定性指标加入到总字典中
            uncertainties["predictive_entropy"].extend(batch_predictive_unc)
            uncertainties["mutual_information"].extend(batch_model_unc)
         

            # 计算预测精度、总样本数和准确率
            correct, total, acc = accuracy(pred_mean, target)

            # 当前批次处理的样本总数
            num_samples = batch_idx * args.batch_size + 1

            # 获取预测类别（取概率最大的类别作为预测值）
            _, preds = torch.max(pred_mean, 1)

            # 计算加权平均精度（balanced accuracy score）
            bacc = balanced_accuracy_score(target.cpu().detach().numpy(),
                                           preds.cpu().detach().numpy())

            # 更新混淆矩阵（统计真实值和预测值的匹配情况）
            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # 更新指标字典，包括损失、准确率等
            metrics.update({
                'correct': correct,  # 当前批次正确预测的样本数
                'total': total,  # 当前批次总样本数
                'loss': loss.item(),  # 当前批次损失
                'accuracy': acc,  # 当前批次准确率
                'bacc': bacc , # 当前批次平衡准确率
                'Predictive Entropy (mean)': np.mean(uncertainties["predictive_entropy"]),
                'Mutual Information (mean)': np.mean(uncertainties["mutual_information"]),
            })

    # 打印验证集的总体统计摘要（包括准确率、损失等）
    print_summary(args, epoch, num_samples, metrics, mode="Validation")

    # 打印不确定性指标的均值，方便分析模型的表现
    print("Predictive Entropy (mean):", np.mean(uncertainties["predictive_entropy"]))
    print("Mutual Information (mean):", np.mean(uncertainties["mutual_information"]))

    # 返回指标对象、混淆矩阵和不确定性字典
    return metrics, confusion_matrix


    
