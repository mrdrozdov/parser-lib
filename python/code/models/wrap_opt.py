import torch
import torch.nn as nn


class WrapOpt(object):
    def __init__(self, context, args):
        super(WrapNet, self).__init__()
        self.opt = context['opt']

    def clip_grad_norm(self):
        pass
