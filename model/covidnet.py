import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module): #Module是nn中定义网络层的基类，所有的网络层都要继承这个基类,是为了网络torch.nn.Sequential中构建一个Flatten层，将输入的多维张量展平为一维张量

    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self, input:torch.Tensor) -> torch.Tensor: #输入是一个张量，输出也是一个张量 其中input：torch.Tensor表示输入是一个张量，-> torch.Tensor表示输出也是一个张量，是提醒开发者输入和输出的数据类型
        
        return input.view(input.size(0),-1)
    

class PEPX(nn.Module):
    """定义 PEPX 卷积神经网络架构。

    该类创建一个 PEPX 模型，其中包含多个卷积层，用于特征投影、扩展、深度表示、第二阶段投影和最终扩展。

    参数:
        n_input (int): 输入特征的数量。我使用in_channel
        n_out (int): 输出特征的数量。我使用out_channel

    属性:
        network (nn.Sequential): 神经网络的层。
    """
    #定义pepx模块，pepx模块包含多个卷积层，用于特征投影、扩展、深度表示、第二阶段投影和最终扩展，特征投影就是降维，特征扩展就是升维，深度表示就是深度卷积，第二阶段投影就是降维，最终扩展就是升维

    def __init__(self, in_channel, out_channel):
       #in_channel是输入的通道数，out_channel是输出的通道数,输入图像的大小是 (batch_size, channels, height, width)
        super(PEPX, self).__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels = in_channel, out_channels= in_channel // 2  , kernel_size=1), #想减少输出的通道数，使用1*1卷积，卷积核的数量就成了输出通道数，想降维或升维都可以使用1*1卷积
            nn.Conv2d(in_channels = in_channel // 2, out_channels= 3 * in_channel // 4, kernel_size=1), #3*3卷积核，padding=1，保持图像大小不变
            nn.Conv2d(in_channels = 3 * in_channel // 4, out_channels= 3 * in_channel // 4, kernel_size=3, groups= 3 * in_channel // 4, padding=1), #groups参数是分组卷积，将输入通道分为groups组，每组有out_channel/groups个通道，每组的卷积核只能看到这一组的输入通道，输出通道数是groups*in_channel/groups = in_channel，实现深度卷积
            nn.Conv2d(in_channels = 3 * in_channel // 4, out_channels= in_channel // 2, kernel_size=1), #1*1卷积核，输出通道数是in_channel//2，逐点卷积
            nn.Conv2d(in_channels = in_channel // 2, out_channels= out_channel, kernel_size=1), #1*1卷积核，输出通道数是out_channel，其中//是整除，向下取整，/是除法返回来是浮点数
            nn.BatchNorm2d(out_channel), #批标准化
  
        )
    
    def forward(self, x):
        """定义前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        return self.network(x)
    
class CovidNet(nn.Module):
    def __init__(self, model, num_classes):
        super(CovidNet, self).__init__()
        
        filters ={ #定义一个字典，存储每个pepx模块的输入和输出通道数,但是这里的输入输出和论文中的图示不一样
            'pepx1_1': [56, 56],
            'pepx1_2': [56, 56],
            'pepx1_3': [56, 56],
            'pepx2_1': [56, 112],
            'pepx2_2': [112, 112],
            'pepx2_3': [112, 112],
            'pepx2_4': [112, 112],
            'pepx3_1': [112, 216],
            'pepx3_2': [216, 216],
            'pepx3_3': [216, 216],
            'pepx3_4': [216, 216],
            'pepx3_5': [216, 216],
            'pepx3_6': [216, 216],
            'pepx4_1': [216, 424],
            'pepx4_2': [424, 424],
            'pepx4_3': [424, 424],

        }

        self.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=56, kernel_size=7, stride=2, padding=3))

        for key in filters: #遍历字典，将pepx模块添加到网络中

          #  if ('pool' in key): # 如果key中包含pool，则添加一个最大池化层
               # self.add_module(key, nn.MaxPool2d(filters[key][0], filters[key][1]))

           # else: 过滤字典并没有pool这个key,所以这个else是多余的
                
            self.add_module(key, PEPX(filters[key][0], filters[key][1]))

        if (model == 'large'):

            self.add_module('conv1_1x1', nn.Conv2d(in_channels=56, out_channels=56, kernel_size=1))
            self.add_module('conv2_1x1', nn.Conv2d(in_channels=56, out_channels=112, kernel_size=1))
            self.add_module('conv3_1x1', nn.Conv2d(in_channels=112, out_channels=216, kernel_size=1))
            self.add_module('conv4_1x1', nn.Conv2d(in_channels=216, out_channels=424, kernel_size=1))

            self.__forward__ = self.forward_large_net  #通过将 self.__forward__ 设置为 self.forward_large_net，代码动态地改变了前向传播方法的行为，使得 forward() 调用时，会使用 forward_large_net 方法，而不是默认的 forward。

        else:
            self.__forward__ = self.forward_small_net

        self.add_module('flatten', Flatten())
        self.add_module('fc1', nn.Linear(7*7*424, 1024))
        self.add_module('Dropout1', nn.Dropout(0.2)) #Dropout(0.2) 表示在训练过程中，每次更新参数时，以 0.2 的概率随机断开 20% 的神经元连接，防止过拟合
        self.add_module('fc2', nn.Linear(1024, 512))
        self.add_module('Dropout2', nn.Dropout(0.2))
        self.add_module('classifier', nn.Linear(512, num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        # 遍历模型中的所有模块
        for m in self.modules():
            # 如果模块是 Conv2d（卷积层）
            if isinstance(m, nn.Conv2d):
                # 使用 He 初始化方法（Kaiming initialization），为卷积层的权重初始化值
                # mode='fan_out' 意味着根据输出的数量初始化权重
                # nonlinearity='relu' 表示使用 ReLU 激活函数的初始化方式
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            # 如果模块是 BatchNorm2d 或 GroupNorm（批标准化层或分组标准化层）
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 为 BatchNorm 或 GroupNorm 层的权重初始化为常数 1
                nn.init.constant_(m.weight, 1)
                # 为 BatchNorm 或 GroupNorm 层的偏置初始化为常数 0
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):

        return self.__forward__(x)
    
    def forward_large_net(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2) #最大池化层，池化核大小是2*2，步长是2，图像大小减半
        out_conv1_1x1 = self.conv1_1x1(x) #进56 出56

        pepx1_1 = self.pepx1_1(x) #进56 出56
        pepx1_2 = self.pepx1_2(pepx1_1 + out_conv1_1x1) #进56 出56
        pepx1_3 = self.pepx1_3(pepx1_2 + pepx1_1+ out_conv1_1x1) #进56 出56

        out_conv2_1x1 = F.max_pool2d(self.conv2_1x1(pepx1_1 + pepx1_2 + pepx1_3 + out_conv1_1x1), 2) #进56 出112

        pepx2_1 = self.pepx2_1( F.max_pool2d(pepx1_1,2) + F.max_pool2d(pepx1_2,2) + F.max_pool2d(pepx1_3,2) + F.max_pool2d(out_conv1_1x1,2)) #进56 出112

        pepx2_2 = self.pepx2_2(pepx2_1 + out_conv2_1x1) #进112 出112
        pepx2_3 = self.pepx2_3(pepx2_2 + pepx2_1 + out_conv2_1x1) #进112 出112
        pepx2_4 = self.pepx2_4(pepx2_3 + pepx2_2 + pepx2_1 + out_conv2_1x1) #进112 出112

        out_conv3_1x1 = F.max_pool2d(self.conv3_1x1(pepx2_1 + pepx2_2 + pepx2_3 + pepx2_4 + out_conv2_1x1), 2) #进112 出216

        pepx3_1 = self.pepx3_1(F.max_pool2d(pepx2_1,2) + F.max_pool2d(pepx2_2,2) + F.max_pool2d(pepx2_3,2) + F.max_pool2d(pepx2_4,2) + F.max_pool2d(out_conv2_1x1,2)) #进112 出216

        pepx3_2 = self.pepx3_2(pepx3_1 + out_conv3_1x1) #进216 出216
        pepx3_3 = self.pepx3_3(pepx3_2 + pepx3_1 + out_conv3_1x1) #进216 出216
        pepx3_4 = self.pepx3_4(pepx3_3 + pepx3_2 + pepx3_1 + out_conv3_1x1) #进216 出216
        pepx3_5 = self.pepx3_5(pepx3_4 + pepx3_3 + pepx3_2 + pepx3_1 + out_conv3_1x1) #进216 出216
        pepx3_6 = self.pepx3_6(pepx3_5 + pepx3_4 + pepx3_3 + pepx3_2 + pepx3_1 + out_conv3_1x1) #进216 出216

        out_conv4_1x1 = F.max_pool2d(self.conv4_1x1(pepx3_1 + pepx3_2 + pepx3_3 + pepx3_4 + pepx3_5 + pepx3_6 + out_conv3_1x1), 2) #进216 出424

        pepx4_1 = self.pepx4_1(F.max_pool2d(pepx3_1,2) + F.max_pool2d(pepx3_2,2) + F.max_pool2d(pepx3_3,2) + F.max_pool2d(pepx3_4,2) + F.max_pool2d(pepx3_5,2) + F.max_pool2d(pepx3_6,2) + F.max_pool2d(out_conv3_1x1,2))
        #进216 出424

        pepx4_2 = self.pepx4_2(pepx4_1 + out_conv4_1x1) #进424 出424
        pepx4_3 = self.pepx4_3(pepx4_2 + pepx4_1 + out_conv4_1x1) # 进424 出424

        flattened = self.flatten(pepx4_3+ pepx4_2 + pepx4_1 + out_conv4_1x1) 

        fc1out = F.relu(self.fc1(flattened))
        #julian 的改动，在全连接层加2个dropout层
        fc1out = self.Dropout1(fc1out)
        fc2out = F.relu(self.fc2(fc1out))
        fc2out = self.Dropout2(fc2out)
        out = self.classifier(fc2out)

        return out
    
    def forward_small_net(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        

        pepx1_1 = self.pepx1_1(x)
        pepx1_2 = self.pepx1_2(pepx1_1) 
        pepx1_3 = self.pepx1_3(pepx1_2 + pepx1_1)

        pepx2_1 = self.pepx2_1( F.max_pool2d(pepx1_1,2) + F.max_pool2d(pepx1_2,2) + F.max_pool2d(pepx1_3,2))
        pepx2_2 = self.pepx2_2(pepx2_1)
        pepx2_3 = self.pepx2_3(pepx2_2 + pepx2_1)
        pepx2_4 = self.pepx2_4(pepx2_3 + pepx2_2 + pepx2_1)


        pepx3_1 = self.pepx3_1(F.max_pool2d(pepx2_1,2) + F.max_pool2d(pepx2_2,2) + F.max_pool2d(pepx2_3,2) + F.max_pool2d(pepx2_4,2))
        pepx3_2 = self.pepx3_2(pepx3_1)
        pepx3_3 = self.pepx3_3(pepx3_2 + pepx3_1)
        pepx3_4 = self.pepx3_4(pepx3_3 + pepx3_2 + pepx3_1)
        pepx3_5 = self.pepx3_5(pepx3_4 + pepx3_3 + pepx3_2 + pepx3_1)
        pepx3_6 = self.pepx3_6(pepx3_5 + pepx3_4 + pepx3_3 + pepx3_2 + pepx3_1)

        pepx4_1 = self.pepx4_1(F.max_pool2d(pepx3_1,2) + F.max_pool2d(pepx3_2,2) + F.max_pool2d(pepx3_3,2) + F.max_pool2d(pepx3_4,2) + F.max_pool2d(pepx3_5,2) + F.max_pool2d(pepx3_6,2))
        pepx4_2 = self.pepx4_2(pepx4_1)
        pepx4_3 = self.pepx4_3(pepx4_2 + pepx4_1)
        flattened = self.flatten(pepx4_3+ pepx4_2 + pepx4_1)

        fc1out = F.relu(self.fc1(flattened))
        #julian 的改动，在全连接层加2个dropout层
        fc1out = self.Dropout1(fc1out)
        fc2out = F.relu(self.fc2(fc1out))
        fc2out = self.Dropout2(fc2out)
        out = self.classifier(fc2out)
        
        return out
    



def covidnet_large(model = "large", num_classes = 3 ,pretrained=False):

    return CovidNet(model = 'large', num_classes = num_classes)

def covidnet_small(model = "small", num_classes = 3 ,pretrained=False):

    return CovidNet(model = 'small', num_classes = num_classes)


