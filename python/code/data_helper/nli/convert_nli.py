import argparse
import json
import os

import nltk


def convert_binary_bracketing(parse):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                tokens.append(word)
                transitions.append(0)

    return tokens, transitions


def build_tree(tokens, transitions):
    stack = []
    buf = tokens[::-1]

    for t in transitions:
        if t == 0:
            stack.append(buf.pop())
        elif t == 1:
            right = stack.pop()
            left = stack.pop()
            stack.append((left, right))

    assert len(stack) == 1

    return stack[0]


class SingleExample(object):
    def __init__(self):
        pass

    @classmethod
    def from_nli(cls, ex):
        pair_id = ex['pairID']

        meta = {}
        meta['group_id'] = pair_id
        meta['label'] = ex['gold_label']

        for i, k in enumerate([1, 2]):
            tokens, transitions = convert_binary_bracketing(
                ex['sentence{}_binary_parse'.format(k)])
            tree = build_tree(tokens, transitions)

            result = cls()
            result.example_id = pair_id + '_{}'.format(i)
            result.tokens = tokens
            result.unlabeled_binary_tree = tree
            result.binary_tree_transitions = transitions
            result.meta = meta
            yield result

    def tojson(self):
        result = {}
        result['example_id'] = self.example_id
        result['tokens'] = self.tokens
        result['unlabeled_binary_tree'] = self.unlabeled_binary_tree
        result['binary_tree_transitions'] = self.binary_tree_transitions
        result['meta'] = self.meta
        return json.dumps(result)


def main(options):

    fout = open(options.output, 'w')

    with open(options.input) as f:
        for line in f:
            for x in SingleExample().from_nli(json.loads(line)):
                fout.write('{}\n'.format(x.tojson()))

    fout.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=os.path.expanduser('~/data/multinli_1.0/multinli_1.0_dev_matched.jsonl'), type=str)
    parser.add_argument('--output', default=os.path.expanduser('~/data/parser_lib/multinli_1.0_dev_matched.jsonl'), type=str)
    options = parser.parse_args()

    main(options)
