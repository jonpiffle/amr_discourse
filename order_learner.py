from __future__ import division

from itertools import combinations

import numpy as np
import numpy.random
import sklearn.linear_model as lm
from simanneal import Annealer

from amr_paragraph import AMRParagraph
from subgraph_learner import generate_paragraphs


def summary(lst):
    return [np.mean(lst), np.std(lst), min(lst), max(lst)]


def overlap(g1, g2):
    return len(set(g1.nodes.keys()) & set(g2.nodes.keys()))


def jaccard(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 0
    return len(s1 & s2) / len(s1 | s2)


def get_features(paragraph, s_graphs):
    amr = paragraph.paragraph_graph()
    node_sets = [set(g.nodes.values()) for g in s_graphs]
    node_jacc = [jaccard(v1, v2) for v1, v2 in zip(node_sets[:-1], node_sets[1:])]

    edge_sets = [g.edges for g in s_graphs]
    E = edge_sets
    jacc = [jaccard(e1, e2) for e1, e2 in zip(E[:-1], E[1:])]
    jacc2 = [jaccard(e1, e2) for e1, e2 in zip(E[:-2], E[2:])]

    combined_jacc = [jaccard(e1 | e2, amr.edges)
                     for e1, e2 in zip(E[:-1], E[1:])]

    feature_vec = []
    feature_vec += summary(jacc)
    feature_vec += summary(jacc2)
    feature_vec += summary(node_jacc)
    feature_vec += summary(combined_jacc)
    #feature_vec += [sum([len(e) * (len(s_graphs) - 1) for e in edge_sets])]
    return feature_vec


def add_negative_examples(paragraphs, k):
    examples, labels = [], []
    for p in paragraphs:
        try:
            p.sentence_graphs()
        except ValueError as e:
            print(e)
            continue
        examples.append(p)
        labels.append(0)
        ordering = np.arange(len(p.amr_sentences))
        reordering = np.arange(len(p.amr_sentences))
        for _ in range(k):
            np.random.shuffle(reordering)
            score = swap_distance(ordering[reordering])
            new_paragraph = build_paragraph_from_existing(p, reordering)
            examples.append(new_paragraph)
            labels.append(score)
    return examples, labels


def build_paragraph_from_existing(old_pgraph, new_order):
    new_pgraph = AMRParagraph(
        old_pgraph.document_name,
        [old_pgraph.amr_sentences[i] for i in new_order],
    )
    s_graphs = old_pgraph.sentence_graphs()
    new_pgraph.s_graphs = [s_graphs[i] for i in new_order]
    new_pgraph.amr_graph = old_pgraph.paragraph_graph()
    return new_pgraph


def swap_distance(order):
    count = 0
    for i, e_i in enumerate(order):
        for j, e_j in enumerate(order[i + 1:]):
            if e_i > e_j:
                count += 1
    return count


if __name__ == '__main__':
    from scorer import OrderScorer
    from optimizer import OrderAnnealer as Orderer


    train = generate_paragraphs('amr.txt', limit=500, k=5)

    examples, labels = add_negative_examples(train, 20)
    n = len(examples)
    weights = n - np.bincount(labels)
    features = np.array([get_features(e, e.sentence_graphs()) for e in examples])

    scorer = OrderScorer()
    #reg = lm.LogisticRegression()
    print('learning')
    scorer.train(features, labels, sample_weight=[weights[i] for i in labels])
    #reg.fit(features, labels)
    print('done')
    test = generate_paragraphs('amr_test.txt', limit=50, k=5)
    good_tests = []
    for t in test:
        try:
            t.sentence_graphs()
            good_tests.append(t)
        except ValueError as e:
            print(e)
            continue
    goodness = []
    for t in good_tests:
        first_order = np.arange(len(t.sentence_graphs()))
        np.random.shuffle(first_order)
        print(first_order)
        orderer = Orderer(first_order, t, t.sentence_graphs(), scorer)
        best, val = orderer.anneal()
        print((best, val))
        goodness.append(swap_distance(best))
    print(summary(goodness), len(goodness))
    test_examples, test_labels = add_negative_examples(good_tests, 20)
    test_features = [get_features(e, e.sentence_graphs()) for e in test_examples]
    """
    predictions = reg.predict(test_features)
    print(reg.score(test_features, test_labels))
    print(reg)
    """
