#!/usr/bin/env python
from __future__ import division

import argparse

import numpy as np
import numpy.random

from amr_paragraph import SlidingWindowGenerator
from order_learner import OrderLearner
from file_parser import FileParser


LEARNERS = {
    'ordering': OrderLearner,
}


def probability(string):
    value = float(string)
    if not (0 <= value <= 1):
        message = '{} must be a probability'.format(value)
        raise argparse.ArgumentTypeError(message)
    return value


def get_learner(**kwargs):
    return LEARNERS[kwargs['learner']]


def main(**kwargs):
    train_data = FileParser().parse(kwargs.pop('train_file'))
    train_window = SlidingWindowGenerator(train_data).generate(k=5)
    # test_data = FileParser().parse(kwargs.pop('test_file'))
    # test_window = SlidingWindowGenerator(test_data).generate(k=5)

    learner = get_learner(**kwargs)(**kwargs)
    train_window, labels = learner.transform(train_window, **kwargs)
    print(train_window)
    print(labels)
    # test_window, labels = learner.transform(test_window, **kwargs)
    learner.train(train_window, labels)
    # learner.evaluate(test_window, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='learn some discourse',
    )

    # TODO: add arguments for different learners
    # no learner specified should train and then run the whole pipeline

    parser.add_argument('-t', '--train_file', default='amr.txt')
    parser.add_argument('-e', '--test_file', default='amr_test.txt')

    subparsers = parser.add_subparsers(
        help='which part of the learning pipeline to run',
        dest='learner',
        metavar='learner',
    )

    subgraph_ordering_parser = subparsers.add_parser('ordering')
    subgraph_ordering_parser.add_argument(
        '-r',
        '--reorder',
        type=probability,
        default=0,
        help='probability of reordering a sentence',
    )

    args = parser.parse_args()
    main(**vars(args))
