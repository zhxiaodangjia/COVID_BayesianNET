import argparse
import torch
import numpy as np
import utils.util as util
from trainer.train_uncetain import initialize, train, train_bayesian, validation, validation_bayesian
#from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau #


def main():
    args = get_arguments()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True #就是使用确定性算法，方便复现
    torch.backends.cudnn.benchmark = False  #保证cudnn不会进行benchmark动态寻找最优的卷积算法，保证每次结果一致
    np.random.seed(SEED)

    args.save = args.save + args.model + '_' + util.datestr() #save the model in a folder with the model name and the date

    if (args.cuda):
        torch.cuda.manual_seed(SEED)

    model, optimizer, training_generator, val_generator, weights, Last_epoch, bflag = initialize(args)  
    #是根据输入参数 返回处理后的不同处理方式，因为他是建立了多个模型，值得借鉴！！
  

    best_pred_loss = 0#the metric is balance accuracy
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-5, verbose=True) #学习率调整策略
    print('Checkpoint folder ', args.save)
    # writer = SummaryWriter(log_dir='../runs/' + args.model, comment=args.model)
    for epoch in range(1, args.nEpochs + 1): #就是根据不同的模式进行训练
        if bflag:
            train_bayesian(args, model, training_generator, optimizer, Last_epoch+epoch, weights)
            val_metrics, confusion_matrix ,= validation_bayesian(args, model, val_generator, Last_epoch+epoch, weights)
        else:
            train(args, model, training_generator, optimizer, Last_epoch+epoch, weights)
            val_metrics, confusion_matrix = validation(args, model, val_generator, Last_epoch+epoch, weights)
       
        BACC = util.BalancedAccuray(confusion_matrix.numpy())
        val_metrics.replace({'bacc': BACC})
        best_pred_loss = util.save_model(model, optimizer, args, val_metrics, Last_epoch+epoch, best_pred_loss, confusion_matrix)

        print(confusion_matrix)
        scheduler.step(val_metrics.avg_loss())

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', default=False,
                        help='load saved_model as initial model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_interval', type=int, default=1000) #每隔多少个batch打印一次
    parser.add_argument('--dataset_name', type=str, default="selfdata")
    parser.add_argument('--nEpochs', type=int, default=24)
    parser.add_argument('--n_monte_carlo', type=int, default=20, help='number of Monte Carlo runs during training')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', default=1e-7, type=float,
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--cuda', action='store_true', default=True)#是否使用cuda!!要改
    #parser.add_argument('--cuda', action='store_true', default=False)#是否使用cuda!!要改
    parser.add_argument('--model', type=str, default='BcovidxNet',
                        choices=('DenseNet','BDenseNet','EfficientNet','BEfficientNet','BcovidxNet'),)
    parser.add_argument('--mode', type=str, default='None',
                        choices=('None','DA'),help='Domain adversarial with respect to the data bases') #是决定是否使用domain adversarial，因为数据来源不同，所以目的是使模型学到的特征对目标域和源域都有效，从而提升跨域泛化能力。
    parser.add_argument('--init_from', action='store_true', default=False, help='In case of Bayessian models, start from a pretrained non Bayesian model')
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--dataset', type=str, default='/home/huaxu@gaps_domain.ssr.upm.es/projects/COVID_BayesianNET/data',
                        help='path to dataset ')
    parser.add_argument('--pre_processing', type=str, default='None',
                        choices=('None','Equalization','CLAHE'))
    parser.add_argument('--saved_model', type=str, default='COVID_BayesianNET/models_saved/Model_best_checkpoint.pth.tar',
                        help='path to save_model ')
    parser.add_argument('--save', type=str, default='COVID_BayesianNET/models_saved/',
                        help='path to checkpoint ')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
