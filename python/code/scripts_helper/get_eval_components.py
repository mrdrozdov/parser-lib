def get_eval_components(context, args, config):
    assert isinstance(config, (list, tuple))

    def helper():
        for cfg in config:
            yield EvalFunc()

    output_context = {
        'eval_funcs': [func for func in helper()]
        }

    return output_context


class EvalFunc(object):

    name = 'default_name'

    def get_batch_iterator(self):
        return [None]
