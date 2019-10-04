import torch
import torch.nn as nn


class SPINN(nn.Module):

    def __init__(self, context, args):
        super(SPINN, self).__init__()

    def forward(self, tokens, transitions=None, run_internal_parser=False, use_internal_parser=False):
        """
        If transitions is None and transition_network exists,
        then transitions will be predicted. Otherwise, exception
        is thrown.
        """
        if transitions is None:
            assert use_internal_parser is True, 'If no transitions provided, then must set use_internal_parser to True.'
        return None
