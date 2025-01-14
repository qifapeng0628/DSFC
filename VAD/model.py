import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from utils import fill_context_mask
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class Model_single(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_single, self).__init__()
        self.fc1 = nn.Linear(n_feature, n_feature) #1408 -> 1408
        self.fc2 = nn.Linear(n_feature*2, n_feature*2) #2816 -> 2816
        self.classifier1 = nn.Linear(n_feature, 1) #1408 -> 1
        self.classifier2 = nn.Linear(n_feature*2, 1) #2816 -> 1
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.75)
        self.DPF = DPF(in_channel=8)
        self.AF = AF(in_channel=1408)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        # [B , T, 3, 1408]
        # -> mean, std, max
        #mean = inputs[:, :, 0, :] #shape=(20,max_len,1408),将原特征的第一行拿出来
        mean = inputs.mean(dim=2)
        r_mean = mean
        r_mean = F.relu(self.fc1(r_mean))
        r1 = torch.tanh(self.classifier1(r_mean)) #shape=(20,max_len,1)
        if is_training:
            r_mean = self.dropout(r_mean)
        mean_score = self.sigmoid(self.classifier1(r_mean)) #shape=(20,max_len,1)
        inputs = inputs.permute(0, 2, 1, 3) # shape = (20,3,max_len,1408)
        channel_ft = self.DPF(inputs)
        channel_ft = torch.squeeze(channel_ft, dim=1)
        #channel_score = F.sigmoid(self.classifier2(r_ft))
        #r2 = F.tanh(self.classifier2(r_ft))
        channel_ft = torch.cat((mean, channel_ft), dim=2)
        all_score = F.relu(self.fc2(channel_ft))
        r2 = torch.tanh(self.classifier2(all_score))
        all_score = self.sigmoid(self.classifier2(all_score))
        score, r = self.AF(mean_score, all_score, r1, r2,)

        return mean_score, all_score, score, r

class AF(nn.Module):

    def __init__(self, in_channel):
        super(AF, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)

    def forward(self, mean, channel, r1, r2):
        x1 = (1-r1*torch.tanh(torch.tensor(1.0)))*mean +r1*torch.tanh(mean)
        x2 = (1-r2*torch.tanh(torch.tensor(1.0)))*x1 +r2*torch.tanh(x1)
        x3 = (1-r1*torch.tanh(torch.tensor(1.0)))*channel + r1*torch.tanh(channel)
        x4 = (1-r2*torch.tanh(torch.tensor(1.0)))*x3 + r2*torch.tanh(x3)
        x5 = x2 * x4
        r = torch.cat([r1, r2], 1)
        return x5, r

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.w = torch.nn.Parameter(w, requires_grad=True)
        self.mix_block = nn.Sigmoid()

    # 前向传播函数，输入两个特征图 fea1 和 fea2
    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


# 定义 Attention 类，基于通道注意力机制的实现
class Attention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)

        self.sigmoid = nn.Sigmoid()

        self.mix = Mix()

    # 前向传播函数
    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return input * out

# ---------------------------------------------------- #
# （2）Spatial attention
class spatial_attention(nn.Module):

    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        padding = kernel_size // 2
        #[b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding)
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):

        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True) #(20,1,max_len,1408)
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True) #(20,1,max_len,1408)
        x = torch.cat([x_maxpool, x_avgpool], dim=1) #(20,2,max_len,1408)
        # [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x) #(20,1,max_len,1408)
        x = self.sigmoid(x)
        outputs = inputs * x #(20,3,max_len,1408)

        return outputs

class DPF(nn.Module):
    def __init__(self, in_channel, kernel_size=7):

        super(DPF, self).__init__()
        self.CAM = Attention(in_channel)
        self.TAM = spatial_attention(kernel_size=kernel_size)
        # In order to keep the shape of the feature map before and after convolution the same, padding is required during convolution
        padding = kernel_size // 2
        # 7*7Convolutional fusion channel information [b,2,h,w]==>[b,1,h,w]
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs= self.CAM(inputs)
        x = self.TAM(inputs)
        x = self.relu(self.conv1(x))
        return x

class Model_single1(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_single1, self).__init__()
        self.fc = nn.Linear(n_feature*2, n_feature*2)
        self.fc1 = nn.Linear(n_feature, 512)
        self.classifier = nn.Linear(n_feature*2, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.70)
        self.DPF = DPF(in_channel=3)
        self.AF = AF(in_channel=3)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        # [B , T, 8, 1408]
        # -> mean, std, 4-max, 2-min
        mean = inputs[:, :, 0, :]
        inputs = inputs.permute(0, 2, 1, 3)
        channel_score = self.DPF(inputs)
        channel_score = torch.squeeze(channel_score, dim=1)
        channel = torch.cat((mean, channel_score), dim=2)
        channel_ = F.relu(self.fc(channel))
        if is_training:
            channel_ = self.dropout(channel_)
        score = self.sigmoid(self.classifier(channel_))
        return mean, score

def model_generater(model_name, feature_size):
    if model_name == 'model_single':
        model = Model_single(feature_size)  # for anomaly detection, only one class, anomaly, is needed.
    elif model_name == 'model_single1':
        model = Model_single1(feature_size)
    else:
        raise ('model_name is out of option')
    return model



