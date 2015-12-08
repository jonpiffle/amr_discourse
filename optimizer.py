import itertools, copy

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
        if strategy == 'greedy':
            pass
        elif strategy == 'anneal':
            pass
        elif strategy == 'baseline':
            pass
        else:
            raise ValueError('incorrect strategy type: %s. Choose from: (greedy, anneal, baseline)' % strategy)


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
