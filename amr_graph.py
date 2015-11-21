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

    def get_parent_edges(self, node, f=lambda e: True):
        return self.get_edges(lambda e: e.out_node == node and f(e))

    def get_edges(self, f):
        return filter(f, self.edges)

    def merge(self, amr):
        """
        Merge the given amr graph into this one. Modifies this graph in place.

        Returns True if the merge was successful, False if the merge could
        not be resolved.
        """
        self.unmerged_nodes = amr.nodes.values()
        while self.unmerged_nodes:
            n = self.unmerged_nodes.pop()
            if n.label in self.nodes:
                self.merge_same_name(amr, self.nodes[n.label], n)
            else:
                self.merge_different_name(amr, n)

    def merge_same_name(self, amr, entity, node):
        f = lambda e: e.label == 'instance'
        world_concept_edge = self.get_parent_edges(entity, f=f)
        amr_concept_edge = amr.get_parent_edges(node, f=f)
        if world_concept_edge and amr_concept_edge:
            # both are instances
            world_concept = world_concept_edge[0].in_node
            amr_concept = amr_concept_edge[0].in_node
            if amr_concept.label == world_concept.label:
                # both are instances of the same concept, so we can merge
                # these nodes without needing to rename
                pass
            else:
                # instances of different concepts, need to rename node not in
                # the world already, but then it can be safely added
                self.find_safe_rename(node.label)
        elif world_concept_edge or amr_concept_edge:
            # one is a concept, the other is an instance. these cannot be
            # merged
            pass
        else:
            # both are concepts. since the names are the same the nodes are too
            # we don't need to add the node to the world, but we do need to
            # bring over any edges
            pass

    def merge_different_name(self, amr, node):
        instance = lambda e: e.label == 'instance'
        amr_concept_edge = amr.get_parent_edges(node, f=instance)
        if amr_concept_edge:
            # :node: is an instance of some concept, we can try to resolve
            # the two graphs
            amr_concept = amr_concept_edge[0]
            # this will look for a concept node in the world graph
            concept_matcher = lambda e: instance(e) and \
                e.in_node.label == amr_concept.label
            world_concept_edge = self.get_edges(concept_matcher)
            if world_concept_edge:
                world_concept = world_concept[0].in_node
                for t in world_concept.edges.keys():
                    to, label = t
                    if label == 'instance':
                        # TODO: check if node attributes match
                        pass
            else:
                # :node: is an instance of a concept which is not in the world
                # we can just add :node: to the world. the concept will be
                # merged in a later iteration
        else:
            # :node: is a concept, since it has a different name, it is a
            # different concept, so we can just add :node: to the world
            pass

    def find_safe_rename(self, label):
        count = 1
        new_name = "{}{}".format(label, count)
        while label in self.nodes:
            count += 1
            new_name = "{}{}".format(label, count)
        return new_name
