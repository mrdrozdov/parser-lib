

class Trainer(object):

    def __init__(self, context, args):
        self.net = context['model_context']['net']
        self.opt = context['train_context'].get('opt', None)
        self.eval_funcs = context['eval_context']['eval_funcs']

    @classmethod
    def build(self, context, args):
        return Trainer(context, args)

    def step(self, batch_context, batch_args):
        pass

    def gradient_update(self):
        pass

    def get_eval_funcs(self):
        return self.eval_funcs
