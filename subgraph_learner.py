import itertools, math, random, time, copy

import numpy as np

from file_parser import FileParser
from amr_paragraph import SlidingWindowGenerator
from partition import Partition
from sklearn.linear_model import LogisticRegression

def union_all(list_of_sets):
    s = set()
    for next_s in list_of_sets:
        s.update(set(next_s))
    return s

def generate_paragraphs(filename, k=3, limit=None):
    entries = FileParser().parse(filename, limit)
    swg = SlidingWindowGenerator(entries)
    paragraphs = swg.generate(k=k)
    return paragraphs

def get_positive_instance(paragraph):
    return set([frozenset(s.edges) for s in paragraph.sentence_graphs()])

def get_initial_partition(paragraph):
    return set([frozenset(paragraph.paragraph_graph().get_subgraph_from_root(root).edges) for root in paragraph.paragraph_graph().get_roots()])

def get_negative_instances(paragraph, target_graph, k=100):
    g = paragraph.paragraph_graph()
    graph_pieces = [g.get_subgraph_from_root(root).edges for root in g.get_roots()]
    p = Partition(graph_pieces)
    m = p.__len__()

    partitions = []
    # k times choose a random partition
    for _ in range(k):
        i = random.randint(0, m-1)
        set_partition = set([frozenset(union_all(list_of_sets)) for list_of_sets in p[i]])

        # need to ensure that the graph is actually a negative instance
        if set_partition != target_graph:
            partitions.append(set_partition)
    return partitions

def generate_instances_and_labels(paragraphs):
    instances, labels = [], []
    for i, paragraph in enumerate(paragraphs):
        try:
            positive_instance = get_positive_instance(paragraph)
        except (ValueError, RuntimeError) as e:
            print(i, e)
            continue
        instances.append(generate_features(paragraph.paragraph_graph(), positive_instance))
        labels.append(1)

        negative_instances = get_negative_instances(paragraph, get_positive_instance(paragraph))
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
    return len(s1 & s2) / float(len(s1 | s2))

def subgraph_similarity(s1, s2):
    s1_full = set(s1.nodes.values()) | s1.edges
    s2_full = set(s2.nodes.values()) | s2.edges
    return jaccard_similarity(s1_full, s2_full)

def generate_features(p_graph, partition):
    features = []

    # number of partitions
    features.append(len(partition)) 

    # mean, min, max, std_dev of #of fragments per partition
    features += summary_statistics([len(p_graph.get_subgraph(s).get_roots()) for s in partition])

    # mean, min, max, std_dev of subgraph similarity for every pair of subgraphs (including a subgraph with itself)
    features += summary_statistics([subgraph_similarity(p_graph.get_subgraph(s1), p_graph.get_subgraph(s2)) for s1, s2 in list(itertools.combinations(partition, 2)) + [(s,s) for s in partition]])

    # mean, min, max, std_dev of verb overlap for every pair of subgraphs (including a subgraph with itself)
    features += summary_statistics([len(p_graph.get_subgraph(s1).get_verbs() & p_graph.get_subgraph(s2).get_verbs()) for s1, s2 in list(itertools.combinations(partition, 2)) + [(s,s) for s in partition]])

    return features

class SearchState(object):
    def __init__(self, p_graph, partition, classifier):
        self.p_graph = p_graph
        self.partition = partition
        self.classifier = classifier

    def get_neighbors(self):
        neighbors = []
        for s1, s2 in itertools.combinations(self.partition, 2):
            partition_copy = copy.copy(self.partition)
            partition_copy.remove(s1)
            partition_copy.remove(s2)
            partition_copy.add(s1|s2)
            neighbors.append(SearchState(self.p_graph, partition_copy, self.classifier))
        return neighbors

    def evaluate(self):
        return self.classifier.predict_proba([generate_features(self.p_graph, self.partition)])[0][1]

def greedy_search(state):
    best, best_val = state, state.evaluate()

    while len(state.get_neighbors()) > 0:
        found_better = False

        for neighbor in state.get_neighbors():
            if neighbor.evaluate() > best_val:
                best, best_val = neighbor, neighbor.evaluate()
                #print(len(best.partition), best_val)
                found_better = True

        if not found_better:
            return best, best_val
        else:
            state = best

train = generate_paragraphs('amr.txt', limit=100)
test = generate_paragraphs('amr_test.txt', limit=10)
instances, labels = generate_instances_and_labels(train)
reg = LogisticRegression(class_weight='auto')
reg.fit(instances, labels)

for t in test:
    try:
        initial_state = SearchState(t.paragraph_graph(), get_initial_partition(t), reg)
    except ValueError:
        continue
    print(greedy_search(initial_state))
    exit()
