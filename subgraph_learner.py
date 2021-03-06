import itertools, math, random, time, copy, os, pickle

import numpy as np

from file_parser import FileParser
from amr_paragraph import SlidingWindowGenerator, AMRParagraph
from partition import Partition
from scorer import SubgraphSelectionScorer
from optimizer import SubgraphOptimizer
from amr_graph import AMRGraph, Edge, Node


class GraphPartitioning(object):
    def __init__(self, p_graph, root_partitioning, subgraph_dict, complex_subgraph_dict):
        self.p_graph = p_graph
        self.root_partitioning = root_partitioning
        self.subgraph_dict = subgraph_dict
        self.complex_subgraph_dict = complex_subgraph_dict
        self.order = None

    def get_subgraph(self, root_set):
        if root_set in self.complex_subgraph_dict:
            return self.complex_subgraph_dict[root_set]

        edge_set = union_all([self.subgraph_dict[root].edges for root in root_set])
        subgraph = self.p_graph.get_subgraph(edge_set)
        self.complex_subgraph_dict[root_set] = subgraph

        return self.complex_subgraph_dict[root_set]

    def get_all_subgraphs(self):
        return [self.get_subgraph(root_set) for root_set in self.root_partitioning]

    def get_ordered_root_sets(self):
        if self.order is None:
            raise ValueError("order attribute must be set before get_ordered_root_sets is called")
        root_list = list(self.root_partitioning)
        return [root_list[i] for i in self.order]

    def copy(self, new_root_partitioning=None):
        if new_root_partitioning is None:
            new_root_partitioning = copy.copy(self.root_partitioning)

        return GraphPartitioning(self.p_graph, new_root_partitioning, self.subgraph_dict, self.complex_subgraph_dict)

    def __eq__(self, other):
        return isinstance(other, GraphPartitioning) and other.root_partitioning == self.root_partitioning

class GraphPartitioningSet(object):
    def __init__(self, p_graph):
        self.p_graph = p_graph
        self.roots = p_graph.get_roots()
        self.subgraph_dict = {root: self.p_graph.get_subgraph_from_root(root) for root in self.roots}
        self.complex_subgraph_dict = {}

        self.all_partitions = Partition(self.roots)
        self.m = self.all_partitions.__len__()

        self.graph_partitionings = {}

    def get_graph_partition(self, k):
        if k in self.graph_partitionings:
            return self.graph_partitionings[k]
        else:
            root_partitioning = set([frozenset(roots) for roots in self.all_partitions[k]])
            graph_partitioning = GraphPartitioning(self.p_graph, root_partitioning, self.subgraph_dict, self.complex_subgraph_dict)
            self.graph_partitionings[k] = graph_partitioning
            return self.graph_partitionings[k]

    def sample_graph_partitionings(self):
        i = random.randint(0, self.m-1)
        return self.get_graph_partition(i)

def union_all(list_of_sets):
    s = set()
    for next_s in list_of_sets:
        s.update(set(next_s))
    return s

def get_positive_instance(paragraph):
    root_partitioning = set([frozenset(s.get_roots()) for s in paragraph.sentence_graphs()])
    subgraph_dict = {}
    complex_subgraph_dict = {}
    for s in paragraph.sentence_graphs():
        edge_set = set()
        roots = s.get_roots()
        for r in roots:
            subgraph_dict[r] = s.get_subgraph_from_root(r)
            edge_set.update(subgraph_dict[r].edges)
        full_subgraph = s.get_subgraph(edge_set)
        complex_subgraph_dict[frozenset(roots)] = full_subgraph

    return GraphPartitioning(None, root_partitioning, subgraph_dict, complex_subgraph_dict)

def get_initial_partition(paragraph):
    roots = paragraph.paragraph_graph().get_roots()
    root_partitioning = set([frozenset([r]) for r in roots])
    subgraph_dict = {r: paragraph.paragraph_graph().get_subgraph_from_root(r) for r in roots}
    return GraphPartitioning(paragraph.paragraph_graph(), root_partitioning, subgraph_dict, {})

def get_negative_instances(paragraph, target_partition, k=100):
    g = paragraph.paragraph_graph()
    partitioning_set = GraphPartitioningSet(g)

    partitions = [partitioning_set.sample_graph_partitionings() for _ in range(k)]
    partitions = [p for p in partitions if p != target_partition]
    return partitions

def generate_instances_and_labels(paragraphs, k=100):
    instances, labels = [], []
    for i, paragraph in enumerate(paragraphs):
        positive_instance = get_positive_instance(paragraph)
        instances.append(generate_features(paragraph.paragraph_graph(), positive_instance))
        labels.append(1)

        negative_instances = get_negative_instances(paragraph, get_positive_instance(paragraph), k=k)
        instances += [generate_features(paragraph.paragraph_graph(), n) for n in negative_instances]
        labels += [0] * len(negative_instances)
    return instances, labels

#### FEATURE FUNCTIONS ####

def mean(lst):
    return sum(lst) / float(len(lst))

def std_dev(lst):
    return math.sqrt(np.var(lst))

def summary_statistics(lst):
    return [mean(lst), std_dev(lst), min(lst), max(lst)]

def jaccard_similarity(s1, s2):
    if len(s1 | s2) == 0:
        return 0
    return len(s1 & s2) / float(len(s1 | s2))

def subgraph_similarity(s1, s2):
    s1_full = set(s1.nodes.values()) | s1.edges
    s2_full = set(s2.nodes.values()) | s2.edges
    return jaccard_similarity(s1_full, s2_full)

def generate_features(p_graph, partition):
    features = []

    # number of partitions
    features.append(len(partition.root_partitioning)) 
    features.append(len(partition.root_partitioning) ** 2) 

    # mean, min, max, std_dev of #of fragments per partition
    features += summary_statistics([len(s) for s in partition.root_partitioning])

    # mean, min, max, std_dev of subgraph similarity for every pair of subgraphs (including a subgraph with itself)
    features += summary_statistics([subgraph_similarity(partition.get_subgraph(s1), partition.get_subgraph(s2)) for s1, s2 in list(itertools.combinations(partition.root_partitioning, 2)) + [(s,s) for s in partition.root_partitioning]])

    # mean, min, max, std_dev of verb overlap for every pair of subgraphs (including a subgraph with itself)
    features += summary_statistics([len(partition.get_subgraph(s1).get_verbs() & partition.get_subgraph(s2).get_verbs()) for s1, s2 in list(itertools.combinations(partition.root_partitioning, 2)) + [(s,s) for s in partition.root_partitioning]])

    return features

if __name__ == '__main__':
    train_instances, train_labels, test_instances, test_labels, test = generate_train_test(use_cache=True)
    scorer = SubgraphSelectionScorer()
    scorer.train(train_instances, train_labels)
    scorer.test(test_instances, test_labels)

    for t in test:
        try:
            optimizer = SubgraphOptimizer(scorer)
            final_state = optimizer.optimize(t)
        except ValueError:
            continue

        print(final_state)

        '''
        final_partition = final_state.partition
        dummy_ordering = list(final_partition.root_partitioning)
        random.shuffle(dummy_ordering)
        actual_ordering = [frozenset(s.get_roots()) for s in t.sentence_graphs()]
        print(dummy_ordering)
        print(actual_ordering)
        print(get_root_swaps(dummy_ordering, actual_ordering))
        '''

        exit()
