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

def get_output_shape(model, image_dim): #确定模型输出的特征尺寸
    feature = model(torch.rand(*image_dim)) # 随机生成一个输入传到模型中，shape = (batch_size, channels, height, width)
    feature = F.adaptive_avg_pool2d(feature, (1, 1)) #自适应平均池化，（1，1）的意思是输出的特征图大小为1*1，所以输出的shape为 (batch_size, num_channels, 1, 1)
    feature = torch.flatten(feature,1) #将特征图展平，shape为(batch_size, num_channels)
    return feature.data.shape[-1] #返回特征的最后一个维度，即特征的长度（num_channels）

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
#这个类是为了实现domain adversarial，即在训练时，将特征图反向传播，使得模型学到的特征对目标域和源域都有效，从而提升跨域泛化能力。
    @staticmethod #静态方法，不需要实例化即可调用
    def forward(ctx, x, alpha):#x为输入，alpha为权重，即反向传播的权重
        ctx.alpha = alpha #将权重保存在ctx中

        return x.view_as(x) #返回输入的形状

    @staticmethod
    def backward(ctx, grad_output): #反向传播
        output = grad_output.neg() * ctx.alpha #将梯度乘以权重，再取反，neg()是取反的函数即是取负号

        return output, None

class Model_DA(torch.nn.Module):
    def __init__(self, model, n_databases):
        super(Model_DA, self).__init__()
        self.base_model = model
        self.domain_classifier = torch.nn.Sequential() #令domain_classifier为一个空的序列
        dim_features = get_output_shape(self.base_model.features,(1, 3, 224, 224)) #获取模型输出特征的长度
        print(f'Dim = {dim_features}')
        self.domain_classifier.add_module('dc_l1',torch.nn.Linear(dim_features, 256))#添加全连接层，输入为特征长度，输出为256
        self.domain_classifier.add_module('dc_l2',torch.nn.Linear(256, n_databases))#添加全连接层，输入为256，输出为数据库数
        #最终 domain_classifier 为一个两层的全连接层
    def forward(self, x):
        if self.training:
            class_output = self.base_model(x) 
            feature = self.base_model.features(x) #获取特征的shape为(batch_size, num_channels, height, width)
            feature = F.relu(feature, inplace=True) #激活函数
            feature = F.adaptive_avg_pool2d(feature, (1, 1)) #自适应平均池化，（1，1）的意思是输出的特征图大小为1*1，所以输出的shape为 (batch_size, num_channels, 1, 1)
            feature = torch.flatten(feature,1) #将特征图展平，shape为(batch_size, num_channels)
            reverse_feature = ReverseLayerF.apply(feature, 1)
             #apply方法是自动调用forward和backward方法，得到的是反向传播的结果，shape为(batch_size, num_channels)
            domain_output = self.domain_classifier(reverse_feature) #将特征传入domain_classifier，得到domain_output
            return class_output, domain_output
        else:
            class_output = self.base_model(x)
            return class_output

    