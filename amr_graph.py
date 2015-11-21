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

    def merge(self, amr):
        """
        Merge the given amr graph into this one. Modifies this graph in place.

        Returns True if the merge was successful, False if the merge could
        not be resolved.
        """
        amr_nodes = amr.nodes.values()
        non_leaf_nodes = set([e.in_node for e in amr.edges])
        leaf_nodes = set(amr_nodes) - non_leaf_nodes
        assert leaf_nodes
        unmerged_nodes = leaf_nodes
        while unmerged_nodes:
            n = unmerged_nodes.pop()
            if n.label in self.nodes:
                # same name
                entity = self.nodes[n.label]
                shared_keys = set(n.attributes.keys()) & \
                    set(entity.attributes.keys())
                attribute_mismatch = [n.attributes[k] != entity.attributes[k]
                                      for k in shared_keys]
                if any(attribute_mismatch):
                    # diff entity
                    new_name = self.find_safe_rename(n.label)
                    n.label = new_name
                    self.add_node(n)
                    for e in n.edges.keys():
                        pass
                else:
                    # same entity
                    # TODO: bring over any new edges
                    for e in n.edges.keys():
                        pass
            else:
                # diff name
                pass

    def find_safe_rename(self, label):
        count = 1
        new_name = "{}{}".format(label, count)
        while label in self.nodes:
            count += 1
            new_name = "{}{}".format(label, count)
        return new_name
