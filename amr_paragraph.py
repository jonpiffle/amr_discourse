import itertools

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
                amr_paragraphs.append(AMRParagraph(document_name, subset))
        return amr_paragraphs


class AMRParagraph(object):
    def __init__(self, document_name, amr_sentences=None):
        if amr_sentences is None:
            amr_sentences = []
        self.amr_sentences = amr_sentences
        self.document_name = document_name


if __name__ == '__main__':
    entries = FileParser().parse('amr.txt')
    swg = SlidingWindowGenerator(entries)
    paragraphs = swg.generate(k=5)
    paragraph = paragraphs[10]
    sentence1 = paragraph.amr_sentences[3]
    sentence2 = paragraph.amr_sentences[4]
    print(sentence1.sentence)
    print(sentence1.amr_graph_string)
    sentence1.amr_graph.draw('g1.gv')
    print(sentence2.sentence)
    print(sentence2.amr_graph_string)
    sentence2.amr_graph.draw('g2.gv')
    sentence1.amr_graph.merge(sentence2.amr_graph)
    sentence1.amr_graph.draw('g3.gv')

