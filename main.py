#!/usr/bin/env python
from __future__ import division

import numpy as np
import numpy.random

from amr_paragraph import SlidingWindowGenerator
from file_parser import FileParser


def get_learner(**kwargs):
    # TODO
    pass


def reorder(sentences):
    ordering = np.shuffle(sentences)
    if ordering == np.range(len(sentences)):
        return sentences, False
    else:
        return sentences[ordering], True


def add_negative_examples(paragraphs, number):
    reorder_prob = len(paragraphs) / number
    examples, labels = [], []
    for p in paragraphs:
        examples.append(p)
        labels.append(1)
        if np.rand() <= reorder_prob:
            sentences, success = reorder(np.array(p.amr_sentences))
            if success:
                new_paragraph = AMRParagraph(
                    p.document_name,
                    sentences,
                )
                examples.append(new_paragraph)
                labels.append(0)
    return np.array(examples), np.array(labels)


def main(**kwargs):
    train_data = FileParser().parse(kwargs.pop('train_file'))
    train_window = SlidingWindowGenerator(train_data).generate(k=5)
    test_data = FileParser().parse(kwargs.pop('test_file'))
    test_window = SlidingWindowGenerator(test_data).generate(k=5)

    learner = get_learner(**kwargs)
    learner.train(train_window)
    learner.evaluate(test_window)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='learn some discourse',
        argument_default=argparse.SUPRESS,
    )

    # TODO: add arguments for different learners
    # no learner specified should train and then run the whole pipeline

    parser.add_argument('-t', '--train_file', default='amr.txt')
    parser.add_argument('-e', '--test_file', default='amr_test.txt')

    args = parser.parse_args()
    main(**vars(args))
