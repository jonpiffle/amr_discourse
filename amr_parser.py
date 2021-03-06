from __future__ import print_function

import re
from collections import deque

from amr_graph import AMRGraph

NON_SPACE_OR_RPAREN = '[^ \n\t\v\)]'
WHITESPACE = re.compile('\s+')
AMR_NODE = re.compile('^\(({}+) / ({}+)(.*)'.format(NON_SPACE_OR_RPAREN,
                                                    NON_SPACE_OR_RPAREN))
AMR_ATTR = re.compile('^:({}+)(.*)'.format(NON_SPACE_OR_RPAREN))
AMR_QUOTE_ATTR_VALUE = re.compile(r'^("[^"]*") ?(.*)')
AMR_REG_ATTR_VALUE = re.compile(r'^([^ ]+) ?(.*)')
CLOSE_PARENS = re.compile('^(\)*).*')


class AMRParser(object):

    def __init__(self):
        self.graph = AMRGraph()
        self.nodes = deque()
        self.eval_later = {}

    def parse(self, amr):
        """
        Builds a DAG from the AMR.

        Senses (watch, boy, etc) never have outgoing edges. Instances of these
        sense have an edge going to the thing that they are an instance of.

        Instances of senses that have arguments also have labeled edges going
        to the instances of each of their arguments.
        """
        tree = self.standardize_spacing(amr)
        while tree:
            if tree[0] == '(':
                node, tree = self.extract_node(tree)
                self.nodes.append(node)
            elif tree[0] == ':':
                tree = self.extract_attr(tree)
            elif tree[0] == ')':
                self.nodes.pop()
                tree = tree[1:].strip()

        return tree

    def extract_node(self, tree):
        m = AMR_NODE.match(tree)
        if not m:
            raise Exception(tree)

        instance, sense, rest = m.groups()
        instance_node, sense_node = [self.graph.add_node(n)
                                     for n in [instance, sense]]
        self.graph.add_edge(instance_node, sense_node, label='instance')

        if instance in self.eval_later:
            # pop the list from the dict, so we don't ever evaluate it again
            later_list = self.eval_later.pop(instance)
            for func, args in later_list:
                func(instance_node, *args)

        return instance_node, rest.strip()

    def extract_attr(self, tree):
        m = AMR_ATTR.match(tree)
        if not m:
            raise Exception(tree)

        attr, rest = m.groups()
        rest = rest.strip()
        if rest[0] == '(':
            n, remainder = self.extract_node(rest)
            self.graph.add_edge(self.nodes[-1], n, label=attr)
            self.nodes.append(n)
        elif rest[0].isalpha():
            # this is a node that has/will be instantiated elsewhere
            span = re.search(r'^\w+\b', rest).span()
            node_name = rest[span[0]:span[1]]
            remainder = rest[span[1]:].strip()
            if node_name not in self.graph.nodes:
                # the node has not actually been instantiated. store a lambda
                # and relevant args to insert the relevant edge whenever the
                # node actually gets instantiated
                later_list = self.eval_later.get(node_name, [])
                later_func = lambda node, other, label: self.graph.add_edge(
                    other,
                    node,
                    label=label,
                )
                later_args = [self.nodes[-1], attr]
                later_list.append((later_func, later_args))
                self.eval_later[node_name] = later_list
            else:
                self.graph.add_edge(
                    self.nodes[-1],
                    self.graph.nodes[node_name],
                    label=attr,
                )
        else:
            if rest[0] == '"':
                val_matcher = AMR_QUOTE_ATTR_VALUE
            else:
                val_matcher = AMR_REG_ATTR_VALUE
            val_match = val_matcher.match(rest)
            if not val_match:
                raise Exception(rest)

            val, remainder = val_match.groups()
            p = CLOSE_PARENS.match(val[::-1])  # strip trailing closed parens
            # put trailing close parens back on the remainder
            if p.groups()[0]:
                val = val[:-len(p.groups()[0])]
                remainder = p.groups()[0] + remainder
            remainder = remainder.strip()
            self.nodes[-1].add_attribute(attr, val)

        return remainder

    def standardize_spacing(self, amr):
        return WHITESPACE.sub(' ', amr)


if __name__ == '__main__':
    import sys

    amr = None
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            amr = f.read()
    else:
        amr = sys.stdin.read()
    print(AMRParser().parse(amr.strip()))
