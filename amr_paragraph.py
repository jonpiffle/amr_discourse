import itertools, copy

from file_parser import FileParser
from amr_graph import AMRGraph

def window(iterable, size):
    """
    Iterates over an iterable size elements at a time
    [1, 2, 3, 4, 5], 3 ->
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    """

    iters = itertools.tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


class SlidingWindowGenerator(object):
    def __init__(self, file_entries):
        self.file_entries = file_entries

    def generate(self, k=5):
        amr_paragraphs = []

        amr_documents = itertools.groupby(self.file_entries, lambda x: x.document_name)
        for document_name, file_entries in amr_documents:
            for subset in window(file_entries, k):
                if not any([s.entry_type != 'body' for s in subset]):
                    amr_paragraphs.append(AMRParagraph(document_name, subset))
        return amr_paragraphs


class AMRParagraph(object):
    def __init__(self, document_name, amr_sentences=None):
        if amr_sentences is None:
            amr_sentences = []
        for s in amr_sentences:
            s.amr_graph = s.amr_graph.deepcopy()
        self.amr_sentences = amr_sentences
        self.document_name = document_name
        self.amr_graph = None
        self.s_graphs = None

    def _generate_amr_graph(self):
        #print(self.amr_sentences[0].entry_id)
        start_graph = AMRGraph()
        self.s_graphs = []
        for sentence in self.amr_sentences:
            #print(sentence.entry_id)
            #sentence_edges = start_graph.merge(sentence.amr_graph.deepcopy())
            sentence_graph = start_graph.merge(sentence.amr_graph)
            self.s_graphs.append(sentence_graph)
        self.amr_graph = start_graph

    def sentence_graphs(self):
        if self.s_graphs is None:
            self._generate_amr_graph()
        return self.s_graphs

    def paragraph_graph(self):
        if self.amr_graph is None:
            self._generate_amr_graph()
        return self.amr_graph

if __name__ == '__main__':
    entries = FileParser().parse('amr.txt', limit=1000)
    swg = SlidingWindowGenerator(entries)
    paragraphs = swg.generate(k=5)
    paragraph = paragraphs[13]
    #paragraph.paragraph_graph().draw()
    print(paragraph.sentence_graphs())
    print(paragraph.amr_sentences[-1].entry_id)
    paragraph.sentence_graphs()[0].draw()
    #print([n.label for n in paragraph.paragraph_graph().get_roots()])

