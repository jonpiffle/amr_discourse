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

    def add_node(self, label):
        """
        Adds a node to the graph, if it's not already in the graph.
        Returns the node.
        """
        self.nodes[label] = self.nodes.get(label, AMRGraph.Node(label))
        return self.nodes[label]
