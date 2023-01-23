import torch
import torchvision
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": True,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
}

def BDenseNet(n_classes=3):

    model = torchvision.models.densenet121(weights='DEFAULT')
    model.classifier = torch.nn.Linear(model.classifier.in_features, n_classes)
    #Turn model into a Bayesian version (in place)
    dnn_to_bnn(model, const_bnn_prior_parameters)

    return model

def DenseNet(n_classes=3):

    model = torchvision.models.densenet121(weights='DEFAULT')
    model.classifier = torch.nn.Linear(model.classifier.in_features, n_classes)
    
    return model

def BEfficientNet(n_classes=3):

    model = torchvision.models.efficientnet_b6(weights='DEFAULT')
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=2304, out_features=n_classes)
        ) 
    #Turn model into a Bayesian version (in place)
    dnn_to_bnn(model, const_bnn_prior_parameters)

    return model

def EfficientNet(n_classes=3):

    model = torchvision.models.efficientnet_b6(weights='DEFAULT')
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=2304, out_features=n_classes)
        ) 

    return model