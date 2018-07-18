import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
class Residual_Model(nn.Module):

	def __init__(self, feature_layers, classifier):
	        super(Residual_Model, self).__init__()
	        self.feature_layers = feature_layers
	        self.classifier = classifier

	def forward(self, input):
		outputs = []
		for layer in self.feature_layers:
			output = layer(input)
			outputs.append(output)
			input = output if len(outputs) is 1 else torch.cat(outputs, 1)
		input = input.view(input.size(0), -1)
		input = self.classifier(input)
		return input


def make_layers(layout):
    layers = []
    # if checkpoint is True:
    #     del layout[0]

    for layer in layout:
        if layer[0] == 'A':
            layers += [nn.AvgPool2d(kernel_size= (layer[1][0], layer[1][1]), stride= layer[2], padding= layer[3])]
        elif layer[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size= (layer[1][0], layer[1][1]), stride= layer[2], padding= layer[3])]
        elif layer[0] == 'C':
            conv2d = nn.Conv2d(in_channels= layer[1], out_channels= layer[2], 
                kernel_size= (layer[3][0], layer[3][1]), stride= layer[4], padding= layer[5])
            if layer[6] == 'ReLU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.ReLU(inplace=True)]
            elif layer[6] == 'PReLU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.PReLU(inplace=True)]
            elif layer[6] == 'SELU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.SELU(inplace=True)]
            elif layer[6] == 'LeakyReLU':
                layers += [conv2d, nn.BatchNorm2d(layer[2]), nn.LeakyReLU(inplace=True)]
            else:
                layers += [conv2d]

    return nn.Sequential(*layers)