import os, pickle, sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

class Scorer(object):
    def __init__(self):
        pass

    def train(self, train_instances, train_labels, use_cache=True):
        pass

    def test(self, test_instances, test_labels):
        pass

    def evaluate(self, test_instance):
        pass


class SubgraphSelectionScorer(Scorer):
    def __init__(self):
        self.classifier = LogisticRegression(class_weight='auto')

    def train(self, train_instances, train_labels, use_cache=True):
        train_filename = 'subgraph_selection_scorer_reg.pickle'
        if use_cache and os.path.exists(train_filename):
            self.classifier = pickle.load(open(train_filename, 'rb'))
        else:
            self.classifier.fit(train_instances, train_labels)
            pickle.dump(self.classifier, open(train_filename, 'wb'))

    def test(self, test_instances, test_labels):
        predictions = self.classifier.predict(test_instances)
        neg_prog, pos_prob = zip(*self.classifier.predict_proba(test_instances))
        print(classification_report(test_labels, predictions))
        print(roc_auc_score(test_labels, pos_prob, 'weighted'))

    def evaluate(self, test_instance):
        return self.classifier.predict_proba([test_instance])[0][1]


class OrderScorer(Scorer):

    def __init__(self): 
        self.classifier = lm.Ridge(alpha=0.1)

    def train(self, train_instances, train_labels, use_cache=True):
        """
        Trains a scorer to score the quality of an ordering of sentences
        Loads from cache if available
        """
        cache = 'subgraph_order_scorer_reg.pickle'
        if use_cache and os.path.exists(cache):
            self.classifier = pickle.load(open(cache, 'rb'))
        else:
            self.classifier.fit(train_instances, train_labels)
            pickle.dump(self.classifier, open(cache, 'wb'))

    def test(self, test_instances, test_labels):
        """ Uses test set to evaluate the performance of the scorer and print it out """
        scores = self.classifier.predict(test_instances)
        # TODO: print report

    def evaluate(self, test_instance):
        """ Applies the scoring function to a given test instance """
        return self.classifier.predict([test_instance])[0]

class PipelineScorer(Scorer):
    def __init__(self):
        pass

    def train(self, train_instances, train_labels, use_cache=True):
        pass

    def test(self, test_instances, test_labels):
        pass

    def evaluate(self, test_instance):
        pass

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

