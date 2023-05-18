import torch
from torch.autograd import Function
import torchvision
import torch.nn.functional as F
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

def get_output_shape(model, image_dim):
    feature = model(torch.rand(*image_dim))
    feature = F.adaptive_avg_pool2d(feature, (1, 1))
    feature = torch.flatten(feature,1)
    return feature.data.shape[-1]

def BDenseNet(n_classes=3, saved_model = ''):

    model = torchvision.models.densenet121(weights='DEFAULT')
    model.classifier = torch.nn.Linear(model.classifier.in_features, n_classes)
    if saved_model:
        checkpoint = torch.load(saved_model,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
    #Turn model into a Bayesian version (in place)
    dnn_to_bnn(model, const_bnn_prior_parameters)

    return model

def DenseNet(n_classes=3):

    model = torchvision.models.densenet121(weights='DEFAULT')
    model.classifier = torch.nn.Linear(model.classifier.in_features, n_classes)
    
    return model

def BEfficientNet(n_classes=3, saved_model = ''):
    model = torchvision.models.efficientnet_b6(weights='DEFAULT')
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=2304, out_features=n_classes)
        ) 
    if saved_model:
        checkpoint = torch.load(saved_model,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
  
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

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Model_DA(torch.nn.Module):
    def __init__(self, model, n_databases):
        super(Model_DA, self).__init__()
        self.model_base = model
        self.domain_classifier = torch.nn.Sequential()
        dim_features = get_output_shape(self.model_base.features,(1, 3, 224, 224))
        print(f'Dim = {dim_features}')
        self.domain_classifier.add_module('dc_l1',torch.nn.Linear(dim_features, 256))
        self.domain_classifier.add_module('dc_l2',torch.nn.Linear(256, n_databases))

    def forward(self, x):
        if self.training:
            class_output = self.model_base(x)
            feature = self.model_base.features(x)
            feature = F.relu(feature, inplace=True)
            feature = F.adaptive_avg_pool2d(feature, (1, 1))
            feature = torch.flatten(feature,1)
            reverse_feature = ReverseLayerF.apply(feature, 1)
            domain_output = self.domain_classifier(reverse_feature)
            return class_output, domain_output
        else:
            class_output = self.model_base(x)
            return class_output

    