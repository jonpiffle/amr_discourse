import itertools, copy

import numpy as np
import numpy.random
from simanneal import Annealer

class Optimizer(object):

    def __init__(self):
        pass

    def optimize(self, initial_state, strategy='greedy'):
        pass 


class SubgraphOptimizer(Optimizer):

    def __init__(self, scorer):
        self.scorer = scorer

    def optimize(self, test_paragraph, strategy='greedy'):
        from subgraph_learner import get_initial_partition, GraphPartitioningSet

        if strategy == 'greedy':
            initial_state = GraphPartitionState(test_paragraph.paragraph_graph(), get_initial_partition(test_paragraph), self.scorer)
            final_state, final_reward = greedy_search(initial_state)
            final_state = final_state.partition
        elif strategy == 'baseline':
            partitioning_set = GraphPartitioningSet(test_paragraph.paragraph_graph())
            final_state = partitioning_set.sample_graph_partitionings()
        else:
            raise ValueError('incorrect strategy type: %s. Choose from: (greedy, baseline)' % strategy)

        return final_state


class OrderOptimizer(Optimizer):

    def __init__(self, scorer):
        self.scorer = scorer

    def optimize(self, graph_partitioning, strategy='greedy'):
        """ Takes a graph_partitioning and an optimization strategy and returns the optimal sentence ordering found """
        pgraph = graph_partitioning.p_graph
        sgraphs = graph_partitioning.get_all_subgraphs()
        if strategy == 'greedy':
            initial_state = SubgraphOrderSearchState(
                pgraph,
                np.arange(len(sgraphs)),
                sgraphs,
                self.scorer,
            )
            opt_order, opt = greedy_search(initial_state)
            opt_order = opt_order.order
        elif strategy == 'anneal':
            annealer = OrderAnnealer(
                np.arange(len(sgraphs)),
                pgraph,
                sgraphs,
                self.scorer,
            )
            opt_order, opt = annealer.anneal()
        elif strategy == 'baseline':
            opt_order = np.arange(len(sgraphs))
            np.random.shuffle(opt_order)
        else:
            raise ValueError('incorrect strategy type: %s. Choose from: (greedy, anneal, baseline)' % strategy)

        graph_partitioning.order = opt_order
        return graph_partitioning


class SearchState(object):

    def __init__(self):
        pass

    def get_neighbors(self):
        pass

    def evaluate(self):
        pass


class GraphPartitionState(SearchState):

    def __init__(self, p_graph, partition, scorer):
        self.p_graph = p_graph
        self.partition = partition
        self.scorer = scorer

    def get_neighbors(self):
        neighbors = []
        root_partitioning = self.partition.root_partitioning
        for s1, s2 in itertools.combinations(root_partitioning, 2):
            root_partition_copy = copy.copy(root_partitioning)
            root_partition_copy.remove(s1)
            root_partition_copy.remove(s2)
            root_partition_copy.add(s1|s2)
            new_partitition = self.partition.copy(root_partition_copy)
            neighbors.append(GraphPartitionState(self.p_graph, new_partitition, self.scorer))
        return neighbors

    def evaluate(self):
        from subgraph_learner import generate_features
        return self.scorer.evaluate(generate_features(self.p_graph, self.partition))


class SubgraphOrderSearchState(SearchState):

    def __init__(self, pgraph, order, sgraphs, scorer):
        self.pgraph = pgraph
        self.order = order
        self.sgraphs = sgraphs
        self.scorer = scorer

    def get_neighbors(self):
        states = []
        for order in self._pairwise_swaps(self.order):
            new_state = SubgraphOrderSearchState(
                self.pgraph,
                order,
                self.sgraphs,
                self.scorer,
            )
            states.append(new_state)
        return states

    def evaluate(self):
        from order_learner import get_features
        sgraphs = [self.sgraphs[i] for i in self.order]
        features = get_features(self.pgraph, sgraphs)
        return self.scorer.evaluate(features)

    def _pairwise_swaps(self, order):
        swaps = []
        for i in range(len(order) - 1):
            new_order = order.copy()
            new_order[i], new_order[i + 1] = order[i + 1], order[i]
            swaps.append(new_order)
        return swaps


def greedy_search(state):
    best, best_val = state, state.evaluate()

    while len(state.get_neighbors()) > 0:
        found_better = False

        for neighbor in state.get_neighbors():
            if neighbor.evaluate() > best_val:
                best, best_val = neighbor, neighbor.evaluate()
                found_better = True

        if not found_better:
            return best, best_val
        else:
            state = best
    return best, best_val


class OrderAnnealer(Annealer):

    steps = 25000

    def __init__(self, initial_state, pgraph, sgraphs, scorer):
        super(OrderAnnealer, self).__init__(initial_state)
        self.scorer = scorer
        self.pgraph = pgraph
        self.sgraphs = sgraphs

    def move(self):
        a, b = np.random.random_integers(0, len(self.state) - 1, size=2)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        from order_learner import get_features
        sgraphs = [self.sgraphs[i] for i in self.state]
        features = get_features(self.pgraph, sgraphs)
        return self.scorer.evaluate(features)

    def update(self, *args, **kwargs):
        pass
