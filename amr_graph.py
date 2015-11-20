class AMRGraph(object):

    class Node(object):

        def __init__(self, label):
            self.label = label
            self.edges = {}
            self.attributes = {}

        def add_edge(self, to, label=None):
            if (to, label) in self.edges:
                return
            self.edges[(to, label)] = AMRGraph.Edge(self, to, label)
            return self.edges[(to, label)]

        def add_attribute(self, attr, value):
            self.attributes[attr] = value

    class Edge(object):
        """Directed, labeled edge"""

        def __init__(self, in_node, out_node, label):
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
            self.edges = self.edges | edge
