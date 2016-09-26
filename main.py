#!/usr/bin/env python
from __future__ import division

import argparse

import numpy as np
import numpy.random
import pickle, os

from amr_paragraph import SlidingWindowGenerator, generate_paragraphs
from file_parser import FileParser
from scorer import SubgraphSelectionScorer, OrderScorer, PipelineScorer
from optimizer import SubgraphOptimizer, OrderOptimizer
from subgraph_learner import generate_instances_and_labels as gen_subgraph_data
from subgraph_learner import GraphPartitioning
from order_learner import generate_instances_and_labels as gen_order_data

def train():
    print('Loading amr data')
    paragraphs = generate_paragraphs('amr.txt', k=5)
    print('%d total cleaned paragraphs' % len(paragraphs))

    print('Training Subgraph Selection Scorer')
    train_instances, train_labels = gen_subgraph_data(paragraphs)
    subgraph_scorer = SubgraphSelectionScorer()
    subgraph_scorer.train(train_instances, train_labels, update_cache=True)

    print('Training Order Scorer')
    train_instances, train_labels, train_weights = gen_order_data(paragraphs)
    order_scorer = OrderScorer()
    order_scorer.train(train_instances, train_labels, train_weights)

    print('Training Pipeline Scorer')
    pipeline_scorer = PipelineScorer()
    subgraph_optimizer = SubgraphOptimizer(subgraph_scorer)
    order_optimizer = OrderOptimizer(order_scorer)
    pipeline_scorer.train(subgraph_optimizer, order_optimizer)

def test():
    print('Loading amr data')
    paragraphs = generate_paragraphs('amr_test.txt', k=5)
    print('%d total cleaned paragraphs' % len(paragraphs))
    paragraphs = paragraphs

    print('Testing Subgraph Selection Scorer')
    test_instances, test_labels = gen_subgraph_data(paragraphs, k=1)
    subgraph_scorer = SubgraphSelectionScorer()
    subgraph_scorer.load()
    subgraph_scorer.test(test_instances, test_labels)

    print('Testing Order Scorer')
    test_instances, test_labels, test_weights = gen_order_data(paragraphs)
    order_scorer = OrderScorer()
    order_scorer.load()
    order_scorer.test(test_instances, test_labels)    

    print('Testing Pipeline Scorer')
    pipeline_scorer = PipelineScorer()
    pipeline_scorer.load()
    baseline_graphs = pipeline_scorer.test(paragraphs, subgraph_strategy='baseline', order_strategy='baseline', processes=3)
    greedy_graphs = pipeline_scorer.test(paragraphs, subgraph_strategy='greedy', order_strategy='greedy', processes=3)
    anneal_graphs = pipeline_scorer.test(paragraphs, subgraph_strategy='greedy', order_strategy='anneal', processes=3)

    pickle.dump(baseline_graphs, open('baseline_graphs.pickle', 'wb'))
    pickle.dump(greedy_graphs, open('greedy_graphs.pickle', 'wb'))
    pickle.dump(anneal_graphs, open('anneal_graphs.pickle', 'wb'))

def example():
    entries = FileParser().parse('paper_example_amr.txt')
    swg = SlidingWindowGenerator(entries)
    paragraphs = swg.generate(k=3)[:1]
    #paragraphs[0].paragraph_graph().draw()
    pipeline_scorer = PipelineScorer()
    pipeline_scorer.load()
    #greedy_graphs = pipeline_scorer.test(paragraphs, subgraph_strategy='greedy', order_strategy='greedy', processes=1)
    #graphs = pipeline_scorer.test(paragraphs, subgraph_strategy='greedy', order_strategy='anneal', processes=1)
    graphs = pipeline_scorer.test(paragraphs, subgraph_strategy='baseline', order_strategy='baseline', processes=1)
    p = graphs[0]
    subgraphs = [p.get_subgraph(r) for r in p.get_ordered_root_sets()]
    print(subgraphs)
    for i, s in enumerate(subgraphs):
        s.draw('random_gv%d' % i)
    #import code; code.interact(local=locals())

LEARNERS = {
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
    #test()
    example()

    '''
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
    '''
