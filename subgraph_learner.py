from file_parser import FileParser
from amr_paragraph import SlidingWindowGenerator
from partition import Partition
import random

def union_all(list_of_sets):
    s = set()
    for next_s in list_of_sets:
        s = s | set(next_s)
    return s

def generate_paragraphs(filename):
    entries = FileParser().parse(filename)
    swg = SlidingWindowGenerator(entries)
    paragraphs = swg.generate(k=2)
    return paragraphs

def get_positive_instances(paragraph):
    return set([frozenset(s) for s in paragraph.sentence_graphs()])

def get_negative_instances(paragraph, target_graph, k=100):
    g = paragraph.paragraph_graph()
    graph_pieces = [g.get_subgraph_from_root(root) for root in g.get_roots()]
    p = Partition(graph_pieces)
    m = len(p)

    partitions = []
    # k times choose a random partition
    for _ in range(k):
        i = random.randint(0, m-1)
        set_partition = set([frozenset(union_all(list_of_sets)) for list_of_sets in p[i]])

        # need to ensure that the graph is actually a negative instance
        if set_partition != target_graph:
            partitions.append(set_partition)
    return partitions

train = generate_paragraphs('amr.txt')
#test = generate_paragraphs('amr_test.txt')

p = train[20]
print(p.amr_sentences[-1].entry_id)
print(len(p.amr_sentences), len(p.paragraph_graph().get_roots()))
target_graph = get_target_graph(p)
print('targ', len(target_graph), [len(s) for s in target_graph])
#p.paragraph_graph().draw()
#for i, s in enumerate(p.amr_sentences):
#    s.amr_graph.draw(filename='%d.gv' % i)
get_train_graphs(p, target_graph)
