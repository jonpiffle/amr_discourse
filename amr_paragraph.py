import copy, itertools

from file_parser import FileParser

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
        self.amr_sentences = amr_sentences
        self.document_name = document_name
        self.amr_graph = None

    def _generate_amr_graph(self):
        start_graph = copy.deepcopy(self.amr_sentences[0].amr_graph)
        for sentence in self.amr_sentences[1:]:
            start_graph.merge(sentence.amr_graph)
        self.amr_graph = start_graph

    def sentence_graphs(self):
        return [s.amr_graph for s in self.amr_sentences]

    def paragraph_graph(self):
        if self.amr_graph is None:
            self._generate_amr_graph()
        return self.amr_graph

if __name__ == '__main__':
    entries = FileParser().parse('amr.txt')
    swg = SlidingWindowGenerator(entries)
    paragraphs = swg.generate(k=5)
    paragraph = paragraphs[17]
    and_sentence = paragraph.sentence_graphs()[-1]
    and_sentence.draw(filename='g3.gv')
    #paragraph.paragraph_graph().draw()
    #print([n.label for n in paragraph.paragraph_graph().get_roots()])

