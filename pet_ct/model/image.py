"""
"""
import torch
import torch.nn as nn
from torchvision.models import Inception3, ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.densenet import _DenseBlock, _DenseLayer, _Transition
import torch.utils.model_zoo as model_zoo



class AvgMerge(nn.Module):
    """
    Averages feature vectors for every item in the batch (one larger "exam" in the
    multi-instance learning setting) and condenses it to a binary softmax probability
    """
    def __init__(self, in_features=2):
        super().__init__()

    def forward(self, input_batch):
        """ Implements the forward pass of average merge.
        Args:
            input (tensor) (batch_size x exam_len x in_features)
        Return:
            output (tensor) batch_size x 2
        """
        a = torch.mean(input_batch, dim=1)
        return a


class AttentionMerge(nn.Module):
    """
    Weighted sum using self attention mechanism.
    """

    def __init__(self, in_features=2, hidden_units=[], dropout_p=0.0, batch_norm=False):
        """
        """
        super().__init__()
        self.in_features = in_features
        self.linear_layers = []
        for out_features in hidden_units:
            self.linear_layers.append(nn.Linear(in_features, out_features))
            self.linear_layers.append(nn.modules.activation.ReLU())
            self.linear_layers.append(nn.Dropout(dropout_p))
            if batch_norm:
                self.linear_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features

        self.linear_layers.append(nn.Linear(in_features, 1))
        self.linear_layers = nn.Sequential(*self.linear_layers)

        self.softmax = nn.modules.activation.Softmax(dim=1)
        self.final_linear_layer = nn.Linear(self.in_features, 2)

    def forward(self, input_batch):
        """ Implements the forward pass of attention merge.
        Args:
            input (tensor) (batch_size x exam_len x in_features)
        Return:
            output (tensor) batch_size x 2
        """
        A = self.linear_layers(input_batch) # returns (batch_size x exam_len x 1)
        A = self.softmax(A) # Each exam's attention weights sum to 1
        input_batch = torch.transpose(input_batch, 2, 1)
        A = torch.bmm(input_batch, A) # returns (batch_size x exam_len x 1)
        A = A.squeeze(2)
        return self.final_linear_layer(A)

model_urls = {
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet_18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


class ClippedInception3(nn.Module):
    """
    Pre-trained densenet with final linear clipped off
    """
    def __init__(self, pretrained=False, out_features=2):
        super().__init__()

        self.net = Inception3(num_classes=1000)
        if pretrained:
            self.net.load_state_dict(model_zoo.load_url(model_urls['inception_v3']))
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, out_features)
        self.out_features = out_features

    def forward(self, input):
        """
        """
        if self.training:
            return self.net(input)[0]
        else:
            return self.net(input)


class ClippedResNet(nn.Module):
    """
    """
    def __init__(self, pretrained=False, out_features=2):
        """
        """
        super().__init__()
        self.net = ResNet(BasicBlock, [2, 2, 2, 2])
        
        if pretrained:
            self.net.load_state_dict(model_zoo.load_url(model_urls['resnet_18']))

        # change last layer 
        self.net.fc = nn.Linear(self.net.fc.in_features, out_features)
        self.out_features = out_features
    
    def forward(self, input):
        """
        """
        return self.net(input)


class ExamModule2D(nn.Module):
    """
    Makes a prediction on an exam using 2D cnns and an aggregation function. An
    exam is an aggregation of images.
    """

    def __init__(self,
                 image_module_class="ClippedResNet", image_module_params={},
                 aggregation_module_class="AvgMerge", aggregation_module_params={},
                 image_module_devices=None):
        """ Each image is fed through the same image_module then the outputs
        of the image module are aggregated together into one prediction using
        the aggregation module. The output of the image_module should match the
        input of the aggregation_module.
        Args:
            image_modules_class (string) name of image_module class
            image_module_parms  (dict)  dict of kwargs for the image_module
            aggregation_module_class (string) name of aggregation_module class
            aggregation_module_params (dict)    dict of kwargs for the aggregation_module
        """
        super().__init__()

        # load image module
        self.image_module = globals()[image_module_class](**image_module_params)
        in_features = self.image_module.out_features

        # load aggregation module
        self.aggregation_module = (globals()[aggregation_module_class]
                                   (in_features=self.image_module.out_features,
                                    **aggregation_module_params))

        # data parallel
        if image_module_devices is not None:
            self.image_module = torch.nn.DataParallel(self.image_module,
                                                      device_ids=image_module_devices)


    def forward(self, input_batch):
        """ The input is passed through the aggregation_module and then
        the image_module.
        Args:
            input (tuple)
                batch_tensor (tensor) batch_size, len_exam, H, W, num_channels
        """

        X = input_batch

        batch_size, len_exam, h, w, c = X.shape

        # flatten batches and exams together
        X = input_batch.view(-1, c, h, w)
        X = self.image_module(X)
        # reassemble exams to shape (B, T, out_features). default: out_features=2
        X = X.view(batch_size, len_exam, -1)

        # aggregate results across T for preds of shape (B, 2)
        X = self.aggregation_module(X)

        return X
