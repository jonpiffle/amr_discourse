#!/usr/bin/env python
from amr_paragraph import SlidingWindowGenerator
from file_parser import FileParser


def get_learner(**kwargs):
    # TODO
    pass


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
