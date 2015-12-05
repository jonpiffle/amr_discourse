from file_parser import FileParser
from amr_paragraph import SlidingWindowGenerator
from partition import Partition
from sklearn.linear_model import LogisticRegression
import random

def union_all(list_of_sets):
    s = set()
    for next_s in list_of_sets:
        s = s | set(next_s)
    return s

def generate_paragraphs(filename, limit=None):
    entries = FileParser().parse(filename, limit)
    swg = SlidingWindowGenerator(entries)
    paragraphs = swg.generate(k=2)
    return paragraphs

def get_positive_instance(paragraph):
    return set([frozenset(s) for s in paragraph.sentence_graphs()])

def get_negative_instances(paragraph, target_graph, k=100):
    g = paragraph.paragraph_graph()
    graph_pieces = [g.get_subgraph_from_root(root) for root in g.get_roots()]
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

def mean(lst):
    return sum(lst) / float(len(lst))

def generate_features(partition):
    return [len(partition), mean([len(s) for s in partition])]

def generate_instances_and_labels(paragraphs):
    instances, labels = [], []
    for i, paragraph in enumerate(paragraphs):
        try:
            positive_instance = get_positive_instance(paragraph)
        except (ValueError, RuntimeError) as e:
            print(i, e)
            continue
        instances.append(generate_features(positive_instance))
        labels.append(1)

        negative_instances = get_negative_instances(paragraph, get_positive_instance(paragraph))
        instances += [generate_features(n) for n in negative_instances]
        labels += [0] * len(negative_instances)
    return instances, labels


train = generate_paragraphs('amr.txt', limit=100)
test = generate_paragraphs('amr_test.txt', limit=10)
instances, labels = generate_instances_and_labels(train)
reg = LogisticRegression(class_weight='auto')
reg.fit(instances, labels)
test_instances, test_labels = generate_instances_and_labels(test)
for proba, label in zip(reg.predict_proba(test_instances), test_labels):
    print(proba, label)
