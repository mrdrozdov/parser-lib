"""

python code/scripts/train.py \
    --input_config '{ "embeddings": { "fmt": "word2vec", "path": "~/data/glove.6B/glove.6B.50d.txt" }}' \
    --model_config '{ "spinn": {}}' \
    --eval_config '{ "parse": {}}' \
    --loss_config '{ "parse": {}}' \
    --data_config '{ "wsj": { "max_token_length": 10 }}' \
    --eval_only_mode

"""

import argparse
import os

from code.models.trainer import Trainer

from code.scripts_helper.get_train_components import get_train_components
from code.scripts_helper.get_eval_components import get_eval_components
from code.scripts_helper.get_data_components import get_data_components
from code.scripts_helper.get_input_components import get_input_components
from code.scripts_helper.get_model_components import get_model_components
from code.scripts_helper.get_loss_components import get_loss_components


def run_eval(trainer):
    for func in trainer.get_eval_funcs():
        print('Running eval = {}'.format(func.name))
        batch_iterator = func.get_batch_iterator()
        for batch in batch_iterator:
            batch_context = {}
            batch_context['batch'] = batch
            batch_args = {}
            batch_args['train'] = False
            trainer.step(batch_context, batch_args)


def run_train():
    pass


def run(options):
    data_context = get_data_components(context={}, args={}, config=options.data_config)
    input_context = get_input_components(context={ 'data_context': data_context }, args={}, config=options.input_config)
    train_context = get_train_components(context={ 'data_context': data_context }, args={}, config=options.train_config)
    eval_context = get_eval_components(context={ 'data_context': data_context }, args={}, config=[options.eval_config])
    model_context = get_model_components(context={}, args={}, config=options.model_config)
    loss_context = get_loss_components(context={}, args={}, config=options.loss_config)

    trainer = Trainer.build(context={
        'train_context': train_context,
        'eval_context': eval_context,
        'input_context': input_context,
        'model_context': model_context,
        'loss_context': loss_context,
        }, args={})

    if options.eval_only_mode:
        run_eval(trainer)
        return

    run_train(trainer)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_config', default=None, type=str)
    parser.add_argument('--model_config', default=None, type=str)
    parser.add_argument('--eval_config', default=None, type=str)
    parser.add_argument('--loss_config', default=None, type=str)
    parser.add_argument('--train_config', default=None, type=str)
    parser.add_argument('--data_config', default=None, type=str, help='If no data is specified for train or eval, then this is used.')
    parser.add_argument('--eval_only_mode', action='store_true')
    options = parser.parse_args()

    run(options)
