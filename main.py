import argparse
import torch
import numpy as np
import utils.util as util
from trainer.train import initialize, train, train_bayesian, validation, validation_bayesian
#from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main():
    args = get_arguments()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    args.save = args.save + args.model + '_' + util.datestr()

    if (args.cuda):
        torch.cuda.manual_seed(SEED)

    model, optimizer, training_generator, val_generator, class_weight, Last_epoch, bflag = initialize(args)

    best_pred_loss = 0#the metric is balance accuracy
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-5, verbose=True)
    print('Checkpoint folder ', args.save)
    # writer = SummaryWriter(log_dir='../runs/' + args.model, comment=args.model)
    for epoch in range(1, args.nEpochs + 1):
        if bflag:
            train_bayesian(args, model, training_generator, optimizer, Last_epoch+epoch, class_weight)
            val_metrics, confusion_matrix = validation_bayesian(args, model, val_generator, Last_epoch+epoch, class_weight)
        else:
            train(args, model, training_generator, optimizer, Last_epoch+epoch, class_weight)
            val_metrics, confusion_matrix = validation(args, model, val_generator, Last_epoch+epoch, class_weight)
       
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
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--dataset_name', type=str, default="COVIDx")
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
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', type=str, default='DenseNet',
                        choices=('DenseNet','BDenseNet','EfficientNet','BEfficientNet'))
    parser.add_argument('--mode', type=str, default='None',
                        choices=('None','DA'),help='Domain adversarial with respect to the data bases')
    parser.add_argument('--init_from', action='store_true', default=False, help='In case of Bayessian models, start from a pretrained non Bayesian model')
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--dataset', type=str, default='COVID_BayesianNET/Data/OrgImagesRescaled/',
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
