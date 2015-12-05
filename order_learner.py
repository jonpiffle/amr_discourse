import numpy as np
import numpy.random


def reorder(sentences):
    ordering = np.shuffle(sentences)
    if ordering == np.range(len(sentences)):
        return sentences, False
    else:
        return sentences[ordering], True


def add_negative_examples(paragraphs, number):
    reorder_prob = len(paragraphs) / number
    examples, labels = [], []
    for p in paragraphs:
        examples.append(p)
        labels.append(1)
        if np.rand() <= reorder_prob:
            sentences, success = reorder(np.array(p.amr_sentences))
            if success:
                new_paragraph = AMRParagraph(
                    p.document_name,
                    sentences,
                )
                examples.append(new_paragraph)
                labels.append(0)
    return np.array(examples), np.array(labels)


class OrderLearner(object):

    def __init__(self, **kwargs):
        pass

    def transform(self, dataset, **kwargs):
        return add_negative_examples(dataset, kwargs['reorder'])

    def train(self, examples, labels):
        pass

    def evaluate(self, examples, labels):
        pass
