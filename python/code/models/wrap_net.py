import torch
import torch.nn as nn


class WrapNet(nn.Module):
    def __init__(self, context, args):
        super(WrapNet, self).__init__()
        self.input_layer = context['input_layer']
        self.core_model = context['core_model']
        self.loss_funcs = context['loss_funcs']

    def forward(self, input_context, input_args):
        pass
