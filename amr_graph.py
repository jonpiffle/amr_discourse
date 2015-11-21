import graphviz as gv
from collections import deque

class AMRGraph(object):

    class Node(object):

        def __init__(self, label):
            self.label = label
            self.edges = {}
            self.attributes = {}

        def add_edge(self, to, label=None):
            if (to, label) in self.edges:
                return self.edges[(to, label)]
            self.edges[(to, label)] = AMRGraph.Edge(self, to, label)
            return self.edges[(to, label)]

        def add_attribute(self, attr, value):
            self.attributes[attr] = value

        def attribute_set(self):
            return set(self.attributes.items())

    class Edge(object):
        """Directed, labeled edge"""

        def __init__(self, out_node, in_node, label):
            self.in_node = in_node
            self.out_node = out_node
            self.label = label

    def __init__(self):
        self.nodes = {}
        self.edges = set()

    def add_node(self, label):
        """
        Adds a node to the graph, if it's not already in the graph.
        Returns the node.
        """
        if isinstance(label, AMRGraph.Node):
            label, node = label.label, label
        else:
            label, node = label, AMRGraph.Node(label)
        self.nodes[label] = self.nodes.get(label, node)
        return self.nodes[label]

    def add_edge(self, from_node, to_node, label=None):
        edge = from_node.add_edge(to_node, label=label)
        if edge is not None:
            self.edges.add(edge)

    def get_parent_edges(self, node, f=lambda e: True):
        return self.get_edges(lambda e: e.in_node == node and f(e))

    def get_child_edges(self, node, f=lambda e: True):
        return self.get_edges(lambda e: e.out_node == node and f(e))

    def get_edges(self, f):
        return list(filter(f, self.edges))

    def get_concept_label(self, entity):
        """
        Returns the concept node associated with an entity by looking for a parent edge
        with label 'instance'. Returns None if no instance edge (this means the entity is a concept).
        Raises error if more than 1 edge matching criteria.
        """
        f = lambda e: e.label == 'instance'
        concept_edges = self.get_child_edges(entity, f=f)
        if len(concept_edges) == 0:
            return None
        elif len(concept_edges) == 1:
            return concept_edges[0].in_node.label
        else:
            raise ValueError('Entity has multiple instance edges')

    def is_concept_node(self, entity):
        return self.get_concept_label(entity) is None

    def merge(self, amr):
        """
        Merge the given amr graph into this one. Modifies this graph in place.

        Returns True if the merge was successful, False if the merge could
        not be resolved.
        """

        rename_map = {}
        self.unmerged_nodes = list(amr.nodes.values())
        while self.unmerged_nodes:
            amr_node = self.unmerged_nodes.pop()
            equiv_nodes = self.find_equiv_nodes(amr, amr_node)

            if len(equiv_nodes) == 0:
                equiv_node = None
            elif len(equiv_nodes) == 1:
                equiv_node = equiv_nodes[0]
            else:
                raise ValueError('Multiple node matches in world graph. Node is ambiguous')

            if equiv_node is None and amr_node.label in self.nodes:
                # No existing node, but a name conflict
                new_name = self.find_safe_rename(amr_node.label)
                rename_map[amr_node.label] = (new_name, 'no existing, but conflict')
                amr_node.label = new_name
                self.add_node(amr_node)
            elif equiv_node is None:
                # No existing node, no name conflict
                self.add_node(amr_node)
                rename_map[amr_node.label] = (amr_node.label, 'no existing, no conflict')
            else:
                # Existing node
                self.merge_node_attributes(equiv_node, amr_node)
                rename_map[amr_node.label] = (equiv_node.label, 'existing')
                amr_node.label = equiv_node.label

        for edge in amr.edges:
            self.add_edge(self.nodes[edge.out_node.label], self.nodes[edge.in_node.label], edge.label)

    def merge_node_attributes(self, node1, node2):
        node1.attributes.update(node2.attributes)

    def find_equiv_nodes(self, amr, amr_node):
        """Returns a list of nodes in world graph that are equiv to node in amr graph"""
        return [n for n in self.nodes.values() if self.nodes_equiv(n, amr, amr_node)]

    def conflicting_attributes(self, node1, node2):
        """
        Returns whether or not node1 and node2 have a different value for an attribute with the same name
        """
        for k, v in node1.attributes.items():
            if k in node2.attributes and node2.attributes[k] != v:
                return True
        return False

    def nodes_equiv(self, self_node, amr, amr_node):
        """
        Two nodes are equivalent if:
            1. they are concept nodes with the same name 
            2. they are the same concepts, they have no conflicting_attributes, and all of their children with the
            same edge label are also equivalent
        """
        if self.is_concept_node(self_node) and amr.is_concept_node(amr_node):
            return self_node.label == amr_node.label 
        elif self.is_concept_node(self_node) or amr.is_concept_node(amr_node):
            return False

        child_pairs = []
        for self_c_edge in self.get_child_edges(self_node):
            for amr_c_edge in amr.get_child_edges(amr_node):
                if self_c_edge.label == amr_c_edge.label:
                    child_pairs.append((self_c_edge.in_node, amr_c_edge.in_node))

        return not self.conflicting_attributes(self_node, amr_node) and all(self.nodes_equiv(self_c, amr, amr_c) for self_c, amr_c in child_pairs)


    def find_safe_rename(self, label):
        count = 1
        new_name = "{}{}".format(label, count)
        while new_name in self.nodes:
            count += 1
            new_name = "{}{}".format(label, count)
        return new_name

    def draw(self, filename='g.gv'):
        def add_nodes(graph, nodes):
            for n in nodes:
                if isinstance(n, tuple):
                    graph.node(n[0], **n[1])
                else:
                    graph.node(n)
            return graph

        def add_edges(graph, edges):
            for e in edges:
                if isinstance(e[0], tuple):
                    graph.edge(*e[0], **e[1])
                else:
                    graph.edge(*e)
            return graph

        g = gv.Digraph()
        nodes = [(n.label, n.attributes) for n in self.nodes.values()]
        edges = [((e.out_node.label, e.in_node.label), {'label': e.label}) for e in self.edges]
        print(nodes)
        g = add_nodes(g, nodes)
        g = add_edges(g, edges)
        g.render('img/%s' % filename, view=True)
