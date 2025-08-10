import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, dropout: float = 0.5): # hidden_size is the size of each hidden layer
        super(CRNN, self).__init__()
        
        #Load resnet18 
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        #resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #takes in grayscale input
        
        #CNN initialization
        self.cnn = nn.Sequential(*list(resnet.children())[:-2]) #removes avgpool & fc
        #Some more stuff to do with cnn
        for name, module in self.cnn.named_modules():
            if name.startswith("layer4.0.conv1") or name.startswith("layer4.0.downsample.0"):
                module.stride = (1,1)


        #RNN initialization
        self.rnn = nn.LSTM(
            input_size = 512,
            hidden_size = hidden_size,
            num_layers = 2,
            bidirectional = True,
            batch_first = True,
            dropout = dropout
        )

        #Final classification
        self.fc = nn.Linear(hidden_size*2, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input (batch of input images): (B, channels=3, height, width)
        features = self.cnn(x) # [B, 512, H_out, W_out]
        features = features.flatten(2).transpose(1, 2)


        #RNN 
        # input: [B, W_out, 512]
        rnn_outp, _ = self.rnn(features)
        output = self.fc(rnn_outp)

        output = output.permute(1, 0, 2)

        # output [W_out, B, num_chars]
        return output