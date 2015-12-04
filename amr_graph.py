import re
import graphviz as gv
from collections import deque

class Node(object):
    nid = 0

    def __init__(self, label, attributes=None):
        self.label = label
        if attributes is None:
            attributes = {}
        self.attributes = attributes
        self.nid = Node.nid
        Node.nid += 1

    def add_attribute(self, attr, value):
        self.attributes[attr] = value

    def __repr__(self):
        return "<Node: %s>" % self.label

    def __eq__(self, other):
        return self.nid == other.nid

    def __hash__(self):
        return hash(self.nid)

class Edge(object):
    """Directed, labeled edge"""

    def __init__(self, out_node, in_node, label):
        self.in_node = in_node
        self.out_node = out_node
        self.label = label

    def __repr__(self):
        return "<Edge: %s -> %s, label: %s>" % (self.out_node.label, self.in_node.label, self.label)

    def __eq__(self, other):
        return self.out_node == other.out_node and self.in_node == other.in_node and self.label == other.label

    def __hash__(self):
        return hash(self.out_node) + hash(self.in_node)

class AMRGraph(object):

    def __init__(self):
        self.nodes = {}
        self.edges = set()

    def add_node(self, label, attributes=None):
        """
        Adds a node to the graph, if it's not already in the graph.
        Returns the node.
        """
        if isinstance(label, Node):
            label, node = label.label, label
        else:
            label, node = label, Node(label, attributes)
        self.nodes[label] = self.nodes.get(label, node)
        return self.nodes[label]

    def add_edge(self, from_node, to_node, label):
        edge = Edge(from_node, to_node, label)
        self.edges.add(edge)
        return edge

    def delete_edge(self, edge):
        self.edges.discard(edge)

    def delete_node(self, node):
        for edge in self.get_child_edges(node) + self.get_parent_edges(node):
            self.delete_edge(edge)

        if node.label in self.nodes:
            del self.nodes[node.label]

    def get_parent_edges(self, node, f=lambda e: True):
        return self.get_edges(lambda e: e.in_node == node and f(e))

    def get_parents(self, node, f=lambda e: True):
        return [e.out_node for e in self.get_parent_edges(node, f)]

    def get_child_edges(self, node, f=lambda e: True):
        return self.get_edges(lambda e: e.out_node == node and f(e))

    def get_children(self, node, f=lambda e: True):
        return [e.in_node for e in self.get_child_edges(node, f)]

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
            print(entity)
            print(self.get_child_edges(entity))
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

        new_edges = []
        for edge in amr.edges:
            new_edge = self.add_edge(self.nodes[edge.out_node.label], self.nodes[edge.in_node.label], edge.label)
            new_edges.append(new_edge)
        return set(new_edges)

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

    '''
    def get_parent_traversals(self, node):
        """
        Given a node, returns a list of 'traversals' (ordered lists of edges) representing all possible paths
        from the current node up the ancestor tree to a root node (a node with no parents)
        Will loop infinitely if there is a loop
        """

        final_traversals = []
        self._get_parent_traversals(node, [[]], final_traversals)
        return final_traversals
    
    def _get_parent_traversals(self, node, traversals, final_traversals):
        """Helper function that does the actual traversal logic"""
        if len(self.get_parent_edges(node)) == 0:
            for t in traversals:
                final_traversals.append(t)
        else:
            for p_edge in self.get_parent_edges(node):
                self._get_parent_traversals(p_edge.out_node, [t + [p_edge] for t in traversals], final_traversals)
    '''

    def get_parent_traversal(self, node):
        visited = set()
        queue = deque([node])
        while len(queue) > 0:
            node = queue.popleft()
            visited.add(node)

            for c in self.get_parent_edges(node):
                if c.out_node not in visited:
                    queue.append(c.out_node)

        edges = [e for e in self.edges if e.in_node in visited and e.out_node in visited]
        return edges

    def remove_and(self):
        while(len(self.get_and_instances()) > 0):
            topological_order = self.topological_sort()
            and_instance = sorted(self.get_and_instances(), key=lambda a: topological_order.index(a))[0]
            traversal = self.get_parent_traversal(and_instance)
            c_edges = self.get_child_edges(and_instance, lambda e: e.label != 'instance')

            for c_edge in c_edges:
                # copy parent traversal
                new_traversal, rename_map, reverse_edge_map = self.copy_traversal(traversal)

                if len(new_traversal) > 0:
                    last_edges = self.get_parent_edges(self.nodes[rename_map[and_instance.label]])
                    for last_edge in last_edges:
                        corresponding_edge = reverse_edge_map[last_edge]
                        # add edge from immediate parent of 'and' instance to child of 'and' instance
                        self.add_edge(last_edge.out_node, c_edge.in_node, label=corresponding_edge.label)
                        # add back other sibling edges
                        for other_edge in self.get_child_edges(corresponding_edge.out_node, lambda e: e.label != corresponding_edge.label):
                            self.add_edge(last_edge.out_node, other_edge.in_node, other_edge.label)
                        # delete extraneous 'and' node (and associated edges)
                        self.delete_node(last_edge.in_node)

            # delete original parent traversal after it has been copied for each child
            for edge in traversal:
                self.delete_edge(edge)
                self.delete_node(edge.out_node)

            # delete all edges from 'and' instance
            for c_edge in c_edges:
                self.delete_edge(c_edge)

            # delete and instance
            self.delete_node(and_instance)

       # delete 'and' node
        if 'and' in self.nodes:
            self.delete_node(self.nodes['and'])

    def is_ancestor_of(self, node1, node2):
        """Returns True is node1 is an ancestor of node 2"""
        traversal = self.get_parent_traversal(node2)
        ancestor_nodes = set([e.in_node for e in traversal]) | set([e.out_node for e in traversal])
        return node1 in nodes

    def topological_sort(self):
        """
        Returns the topological ordering of the nodes; see: https://en.wikipedia.org/wiki/Topological_sorting#Algorithms.
        """
        L = []
        E = set(self.edges)
        S = self.get_roots()
        while len(S) > 0:
            n = S.pop()
            L.append(n)
            c_edges = [e for e in E if e.out_node == n]
            for c_edge in c_edges:
                E.remove(c_edge)
                m = c_edge.in_node
                if len([e for e in E if e.in_node == m]) == 0:
                    S.append(m)
        return L

    def copy_traversal(self, traversal):
        new_traversal = []
        rename_map = {}
        reverse_edge_map = {}

        for edge in traversal:
            if edge.out_node.label not in rename_map:
                out_node = self.add_node(self.find_safe_rename(edge.out_node.label), edge.out_node.attributes)
                rename_map[edge.out_node.label] = out_node.label
            else:
                out_node = self.nodes[rename_map[edge.out_node.label]]

            if edge.in_node.label not in rename_map:
                in_node = self.add_node(self.find_safe_rename(edge.in_node.label), edge.in_node.attributes)
                rename_map[edge.in_node.label] = in_node.label
            else:
                in_node = self.nodes[rename_map[edge.in_node.label]]

            new_edge = self.add_edge(out_node, in_node, edge.label)
            reverse_edge_map[new_edge] = edge
            new_traversal.append(new_edge)

            # copy other edges along traversal; however, point to original child nodes
            for other_edge in self.get_child_edges(edge.out_node, lambda e: e != edge):
                new_edge = self.add_edge(out_node, other_edge.in_node, other_edge.label)

        return new_traversal, rename_map, reverse_edge_map

    def get_and_instances(self):
        and_nodes = [n for n in self.nodes.values() if n.label == 'and']
        if len(and_nodes) == 0:
            return []

        and_node = and_nodes[0]
        and_instances = [e.out_node for e in self.get_parent_edges(and_node, lambda e: e.label == 'instance')]
        return and_instances

    def get_roots(self):
        return [n for n in self.nodes.values() if len(self.get_parent_edges(n)) == 0]

    def get_verbs(self):
        return [n for n in self.nodes.values() if re.search('-\d+', n.label) is not None]

    def get_verb_instances(self):
        v_instances = [n for n in self.nodes.values() if self.get_concept_label(n) in [v.label for v in self.get_verbs()]]
        v_instances_with_args = [v for v in v_instances if len(self.get_child_edges(v, lambda e: 'ARG' in e.label)) > 0]
        return v_instances_with_args

    def get_subgraph_from_root(self, root):
        visited = set()
        queue = deque([root])
        while len(queue) > 0:
            node = queue.popleft()
            visited.add(node)

            for c in self.get_child_edges(node):
                if c.in_node not in visited:
                    queue.append(c.in_node)

        edges = [e for e in self.edges if e.in_node in visited and e.out_node in visited]
        return edges

    def reverse_arg_ofs(self):
        for e in self.edges:
            if re.match('ARG\d-of', e.label) is not None:
                e.label = e.label.replace('-of', '')
                e.out_node, e.in_node = e.in_node, e.out_node

    def find_safe_rename(self, label):
        count = 1
        new_name = "{}{}".format(label, count)
        while new_name in self.nodes:
            count += 1
            new_name = "{}{}".format(label, count)
        return new_name

    def deepcopy(self):
        new_graph = AMRGraph()

        node_map = {}
        for n in self.nodes.values():
            new_node = new_graph.add_node(n.label, n.attributes)
            node_map[n] = new_node

        for e in self.edges:
            new_graph.add_edge(node_map[e.out_node], node_map[e.in_node], e.label)

        return new_graph

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
        g = add_nodes(g, nodes)
        g = add_edges(g, edges)
        g.render('img/%s' % filename, view=True)
