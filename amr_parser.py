from __future__ import print_function

import re
from collections import deque

from amr_graph import AMRGraph

NON_SPACE_OR_RPAREN = '[^ \n\t\v\)]'
WHITESPACE = re.compile('\s+')
AMR_NODE = re.compile('^\(({}+) / ({}+)(.*)'.format(NON_SPACE_OR_RPAREN,
                                                    NON_SPACE_OR_RPAREN))
AMR_ATTR = re.compile('^:({}+)(.*)'.format(NON_SPACE_OR_RPAREN))
AMR_ATTR_VALUE = re.compile('^({}+)(.*)'.format(NON_SPACE_OR_RPAREN))

class AMRParser(object):

    def __init__(self):
        self.graph = AMRGraph()
        self.nodes = deque()

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
        instance_node.add_edge(sense_node, label='instance')

        return instance_node, rest.strip()

    def extract_attr(self, tree):
        m = AMR_ATTR.match(tree)
        if not m:
            raise Exception(tree)

        attr, rest = m.groups()
        rest = rest.strip()
        if rest[0] == '(':
            n, remainder = self.extract_node(rest)
            self.nodes[-1].add_edge(n, label=attr)
            self.nodes.append(n)
        else:
            val_match = AMR_ATTR_VALUE.match(rest)
            if not val_match:
                raise Exception(rest)

            val, remainder = val_match.groups()
            remainder = remainder.strip()
            self.nodes[-1].add_attribute(attr, val)

        return remainder

    def standardize_spacing(self, amr):
        """Remove extraneous whitespace characters."""
        return WHITESPACE.sub(' ', amr)


if __name__ == '__main__':
    import sys

    # amr = sys.stdin.read()
    amr = """
(s / say-01
  :ARG0 (u / university
          :name (n / name
                  :op1 "Naif"
                  :op2 "Arab"
                  :op3 "Academy"
                  :op4 "for"
                  :op5 "Security"
                  :op6 "Sciences")
          :ARG1-of (b / base-01
                     :location (c / city
                                 :name (n2 / name
                                         :op1 "Riyadh"))))
  :ARG1 (r / run-01
          :ARG0 u
          :ARG1 (w / workshop
                  :beneficiary (p / person
                                 :quant 50
                                 :ARG1-of (e / expert-41
                                            :ARG2 (c2 / counter-01
                                                    :ARG1 (t2 / terrorism)))))
          :duration (t / temporal-quantity
                      :quant 2
                      :unit (w2 / week)))
  :medium (s2 / statement))
        """

    amr2 = """
(a0 / watch
      :ARG0 (a1 / boy)
      :ARG1 (a2 / tv))
"""
    print(AMRParser().parse(amr.strip()))
