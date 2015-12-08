import os, pickle, sys

import numpy as np

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, classification_report

def summary(lst):
    # set features to 0 if empty lst (like cases where list is too small to get 2-step similarity)
    if len(lst) == 0:
        return [0, 0, 0, 0]

    return [np.mean(lst), np.std(lst), min(lst), max(lst)]

class Scorer(object):
    def __init__(self):
        pass

    def train(self, train_instances, train_labels, update_cache=True):
        pass

    def test(self, test_instances, test_labels):
        pass

    def load(self):
        pass

    def evaluate(self, test_instance):
        pass


class SubgraphSelectionScorer(Scorer):
    def __init__(self):
        self.classifier = LogisticRegression(class_weight='auto')
        self.cache_filename = 'subgraph_selection_scorer_reg.pickle'

    def train(self, train_instances, train_labels, update_cache=True):
        self.classifier.fit(train_instances, train_labels)
        if update_cache:
            pickle.dump(self.classifier, open(self.cache_filename, 'wb'))

    def load(self):
        if os.path.exists(self.cache_filename):
            self.classifier = pickle.load(open(self.cache_filename, 'rb'))
        else:
            raise Exception("No classifier exists! Must call train with update_cache=True")

    def test(self, test_instances, test_labels):
        predictions = self.classifier.predict(test_instances)
        neg_prog, pos_prob = zip(*self.classifier.predict_proba(test_instances))
        print(classification_report(test_labels, predictions))
        print(roc_auc_score(test_labels, pos_prob, 'weighted'))

    def evaluate(self, test_instance):
        return self.classifier.predict_proba([test_instance])[0][1]


class OrderScorer(Scorer):

    def __init__(self):
        self.classifier = Ridge(alpha=0.1)
        self.cache_filename = 'subgraph_order_scorer_reg.pickle'

    def train(self, train_instances, train_labels, update_cache=True,
              sample_weight=None):
        """
        Trains a scorer to score the quality of an ordering of sentences
        Loads from cache if available
        """
        self.classifier.fit(train_instances, train_labels, sample_weight=sample_weight)
        if update_cache:
            pickle.dump(self.classifier, open(self.cache_filename, 'wb'))

    def test(self, test_instances, test_labels):
        """ Uses test set to evaluate the performance of the scorer and print it out """
        scores = self.classifier.predict(test_instances)
        # TODO: print report

    def load(self):
        if os.path.exists(self.cache_filename):
            self.classifier = pickle.load(open(self.cache_filename, 'rb'))
        else:
            raise Exception("No classifier exists! Must call train with update_cache=True") 

    def evaluate(self, test_instance):
        """ Applies the scoring function to a given test instance """
        return self.classifier.predict([test_instance])[0]

class PipelineScorer(Scorer):
    def __init__(self):
        self.subgraph_optimizer = None
        self.order_optimizer = None
        self.cache_filename = 'pipeline_scorer_reg.pickle'

    def train(self, subgraph_optimizer, order_optimizer, update_cache=True):
        self.subgraph_optimizer = subgraph_optimizer
        self.order_optimizer = order_optimizer
        if update_cache:
            pickle.dump((self.subgraph_optimizer, self.order_optimizer), open(self.cache_filename, 'wb'))

    def test(self, test_paragraphs, subgraph_strategy='greedy', order_strategy='greedy'):
        print('subgraph_strategy: %s' % subgraph_strategy)
        print('order_strategy: %s' % order_strategy)
        print('mean, std_dev, min, max kendall tau: ', 
              summary([self.evaluate(test_paragraph, subgraph_strategy, order_strategy) for test_paragraph in test_paragraphs]))
        print()

    def load(self):
        if os.path.exists(self.cache_filename):
            self.subgraph_optimizer, self.order_optimizer = pickle.load(open(self.cache_filename, 'rb'))
        else:
            raise Exception("No classifier exists! Must call train with update_cache=True") 

    def evaluate(self, test_paragraph, subgraph_strategy='greedy', order_strategy='greedy'):
        best_graph_partition = self.subgraph_optimizer.optimize(test_paragraph, subgraph_strategy)
        best_graph_partition_with_order = self.order_optimizer.optimize(best_graph_partition, order_strategy)
        target_root_ordering = [frozenset(s.get_roots()) for s in test_paragraph.sentence_graphs()]
        test_root_ordering = best_graph_partition_with_order.get_ordered_root_sets()
        root_swaps = get_root_swaps(test_root_ordering, target_root_ordering)
        print(subgraph_strategy, order_strategy, root_swaps)
        return root_swaps

def get_root_swaps(root_partitions, goal_root_partitions):
    def get_goal_partition_index(root):
        return next(i for i, root_partition in enumerate(goal_root_partitions) if root in root_partition)

    count = 0
    for i, root_partition_i in enumerate(root_partitions):
        for root_partition_j in root_partitions[i + 1:]:
            for root_i in root_partition_i:
                for root_j in root_partition_j:
                    if get_goal_partition_index(root_i) > get_goal_partition_index(root_j):
                        count += 1
    return count

if __name__ == '__main__':
    from subgraph_learner import generate_train_test, generate_paragraphs
    from order_learner import get_features, add_negative_examples
    from optimizer import SubgraphOptimizer, OrderOptimizer

    train_instances, train_labels, test_instances, test_labels, test = generate_train_test(use_cache=False)
    subgraph_scorer = SubgraphSelectionScorer()
    subgraph_scorer.load()
    subgraph_optimizer = SubgraphOptimizer(subgraph_scorer)

    #train = generate_paragraphs('amr.txt', limit=500, k=5)
    #examples, labels = add_negative_examples(train, 20)
    #n = len(examples)
    #weights = n - np.bincount(labels)
    #features = np.array([get_features(e, e.sentence_graphs()) for e in examples])
    order_scorer = OrderScorer()
    order_scorer.load()
    #order_scorer.train(features, labels, sample_weight=[weights[i] for i in labels])
    order_optimizer = OrderOptimizer(order_scorer)

    pipeline_scorer = PipelineScorer()
    pipeline_scorer.train(subgraph_optimizer, order_optimizer)
    test = test[:30]

    pipeline_scorer.test(test, subgraph_strategy='baseline', order_strategy='baseline')
    #pipeline_scorer.test(cleaned_test, subgraph_strategy='greedy', order_strategy='greedy')
    #pipeline_scorer.test(cleaned_test, subgraph_strategy='greedy', order_strategy='anneal')
    #print(pipeline_scorer.evaluate(t, subgraph_strategy='baseline', order_strategy='baseline'))
    #print(pipeline_scorer.evaluate(t, subgraph_strategy='greedy', order_strategy='greedy'))
    #print(pipeline_scorer.evaluate(t, subgraph_strategy='greedy', order_strategy='anneal'))
