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
		for layer in enumerate(self.feature_layers):
			output = layer(input)
			if layer isinstance(nn.Conv2d):
				outputs.append(output)
				input = torch.cat(outputs, 1)
			else:
				input = output
		input = input.view(input.size(0), -1)
		input = self.classifier(input)
		return input
