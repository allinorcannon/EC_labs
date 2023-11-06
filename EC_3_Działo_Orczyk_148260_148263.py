import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Tuple
import copy
from copy import deepcopy
from itertools import combinations, product
import time
import numpy as np
from joblib import Parallel, delayed

MAX_DIST = 2147483647
random.seed(100)

def create_distance_matrix(df):
    x = df['x'].values
    y = df['y'].values
    x1 = x.reshape((df.shape[0], 1))
    x2 = x.reshape((1, df.shape[0]))
    y1 = y.reshape((df.shape[0], 1))
    y2 = y.reshape((1, df.shape[0]))
    matrix = np.round(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)).astype(int)
    np.fill_diagonal(matrix, MAX_DIST)
    return matrix

def evaluate(distance_matrix: np.ndarray, path: np.array, costs: np.array) -> int:
    copy_path = copy.deepcopy(path)
    copy_path = np.append(copy_path, copy_path[0])
    distances = distance_matrix[copy_path[:-1], copy_path[1:]]
    path_costs = costs[copy_path[:-1]]
    total_length = np.sum(distances + path_costs)
    return total_length

def random_sequence(distance_matrix: np.ndarray, start_node: int = 0) -> List[int]:
    n = distance_matrix.shape[0]
    num_selected = n // 2 if n % 2 == 0 else (n // 2) + 1
    path = random.sample(list(range(distance_matrix.shape[0])), num_selected)
    return np.array(path)

class Delta:
  def __init__(self, delta_value: int, delta_type: str, values: tuple):
      self.delta_value = delta_value
      self.delta_type = delta_type
      self.values = values
  
  def create_path(self, path):
      if self.delta_type == 'inter':
          index_of_old_node, new_value = self.values
          modified_path = copy.deepcopy(path)
          modified_path[index_of_old_node] = new_value
          return modified_path
      elif self.delta_type == 'edges_intra':
          index1, index2 = self.values
          if index2 == 0:
              modified_path = np.array(list(path[:index1+1]) + list(path[index2-1:index1:-1]))
          else:
              modified_path = np.array(list(path[:index1+1]) + list(path[index2-1:index1:-1]) + list(path[index2:]))
          return modified_path
      elif self.delta_type == 'nodes_intra':
          index1, index2 = self.values
          modified_path = copy.deepcopy(path)
          modified_path[index1] = path[index2]
          modified_path[index2] = path[index1]
          return modified_path
      else:
          raise ValueError("Invalid delta_type!")

def inter(path, distance_matrix, costs, generator_type: str = 'steepest'):
    og_nodes = set(path)
    nodes = set(list(range(distance_matrix.shape[0])))
    changes = product(og_nodes, nodes - og_nodes)

    if generator_type == 'greedy':
        changes = list(changes)

        random.shuffle(changes)

    node_indices = {node: index for index, node in enumerate(path)}

    neighbourhood_deltas = []

    for change in changes:
        old_node, new_node = change
        delta_value = 0
        index_of_old_node = node_indices[old_node]

        prev = path[(index_of_old_node - 1) % len(path)]
        next = path[(index_of_old_node + 1) % len(path)]

        old_dist = distance_matrix[prev, old_node] \
                  + distance_matrix[old_node, next]
        new_dist = distance_matrix[prev, new_node] \
                  + distance_matrix[new_node, next]
        delta_value += new_dist - old_dist  # lower is better

        delta_value += costs[new_node] - costs[old_node]  # lower is better

        delta = Delta(delta_value, 'inter', (index_of_old_node, new_node))

        if generator_type == 'greedy' and delta_value < 0:
            return [delta]

        neighbourhood_deltas.append(delta)
    if generator_type == 'greedy':
        return []
    return neighbourhood_deltas

def edges_intra(path, distance_matrix, generator_type: str = 'steepest'):
    pairs1 = np.column_stack((path[:-1], path[1:]))

    cyclic_pair = np.array([path[-1], path[0]])

    all_edges = np.vstack((pairs1, cyclic_pair))

    edge_swaps = combinations(all_edges, 2)

    if generator_type == 'greedy':
        edge_swaps = list(edge_swaps)

        random.shuffle(edge_swaps)

    neighbourhood_deltas = []

    node_indices = {node: index for index, node in enumerate(path)}
    # remove non valid edge swaps
    for swap in edge_swaps:
        edge1, edge2 = swap
        if edge1[1] == edge2[0] or edge1[0] == edge2[1]:
            continue

        # swapping:
        # given (2,3) and (3, 4, 5, 6) and (6,7) (edge1[0], edge1[1]) and (edge2[0], edge2[1])
        # result is (2,6) and (6, 5, 4, 3) and (3,7) (edge1[0], edge2[0]) and (edge1[1], edge2[1]) + reverse all inbetween
        # so basically just reversing path from point x to point y
        # delta is new distance - old distance
        delta_value = (
            distance_matrix[edge1[0], edge2[0]]
            + distance_matrix[edge1[1], edge2[1]]
        ) - (
            distance_matrix[edge1[0], edge1[1]]
            + distance_matrix[edge2[1], edge2[0]]
        )
        index1 = node_indices[edge1[0]]
        index2 = node_indices[edge2[1]]

        delta = Delta(delta_value, 'edges_intra', (index1, index2))

        if generator_type == 'greedy' and delta_value < 0:
            return [delta]

        neighbourhood_deltas.append(delta)

    # Size: n*(n - 3)/2
    if generator_type == 'greedy':
        return []
    return neighbourhood_deltas

def nodes_intra(path: np.array, distance_matrix: np.array, generator_type: str = 'steepest'):

    changes = combinations(path, 2)
    neighbourhood_deltas = []

    if generator_type == 'greedy':
        changes = list(changes)

        random.shuffle(changes)

    node_indices = {node: index for index, node in enumerate(path)}

    for change in changes:
        delta_value = 0
        # 1 2 3 4 5 6 7 8 9
        # exchange 3 and 7
        # need to calculate (2,7) dist[path[node1-1], path[node2]]
        # and (7,4) and so on
        node1, node2 = change
        index1 = node_indices[node1]
        index2 = node_indices[node2]

        prev_index1 = (index1 - 1) % len(path)
        prev_index2 = (index2 - 1) % len(path)
        next_index1 = (index1 + 1) % len(path)
        next_index2 = (index2 + 1) % len(path)

        #figure out a better way to do this if needed
        modified_path = copy.deepcopy(path)
        modified_path[index1] = path[index2]
        modified_path[index2] = path[index1]

        old_dist = distance_matrix[path[prev_index1], path[index1]] + distance_matrix[path[prev_index2], path[index2]]
        new_dist = distance_matrix[modified_path[prev_index1], modified_path[index1]] + distance_matrix[modified_path[prev_index2], modified_path[index2]]

        delta_value += new_dist - old_dist

        old_dist = distance_matrix[path[index1], path[next_index1]] + distance_matrix[path[index2], path[next_index2]]
        new_dist = distance_matrix[modified_path[index1], modified_path[next_index1]] + distance_matrix[modified_path[index2], modified_path[next_index2]]

        delta_value += new_dist - old_dist

        delta = Delta(delta_value, 'nodes_intra', (index1, index2))

        if generator_type == 'greedy' and delta_value < 0:
            return [delta]

        neighbourhood_deltas.append(delta)

    # Size: n*(n - 1)/2
    if generator_type == 'greedy':
        return []
    return neighbourhood_deltas

def intra(path, intra_type, distance_matrix, search_type):
    if intra_type == "nodes":
        return nodes_intra(path, distance_matrix, search_type)
    elif intra_type == "edges":
        return edges_intra(path, distance_matrix, search_type)
    else:
        raise ValueError("Invalid intra_type")

def local_search(initial_solution, costs, distance_matrix, search_type, intra_type):
    if initial_solution is None:
        initial_solution = random_sequence(distance_matrix)

    initial_evaluation = evaluate(distance_matrix, initial_solution, costs)
    current_path = initial_solution
    i = 0
    while True:
        i += 1

        neighbourhood_inter_deltas = inter(current_path, distance_matrix, costs, search_type)
        neighbourhood_intra_deltas = intra(current_path, intra_type, distance_matrix, search_type)
        neighbourhood_deltas = np.array(neighbourhood_inter_deltas + neighbourhood_intra_deltas)
        if search_type == "steepest":
            min_index = np.argmin([delta.delta_value for delta in neighbourhood_deltas])
            best_delta = neighbourhood_deltas[min_index]

            if (best_delta.delta_value) < 0:
                current_path = best_delta.create_path(current_path)
            else:
                # No negative deltas found, break the loop
                break

        elif search_type == "greedy":
            if neighbourhood_deltas.size > 0:
                best_delta = np.random.choice(neighbourhood_deltas)
                current_path = best_delta.create_path(current_path)
            else:
                # No negative deltas found, break the loop
                break

        else:
            raise ValueError("Invalid search_type")

    final_evaluation = evaluate(distance_matrix, current_path, costs)
    return initial_evaluation, current_path, final_evaluation

def evaluate_solution(filename, initial, version, intra_type, costs, distance_matrix):
    start = time.time()
    initial_score, path, score = local_search(initial, costs, distance_matrix, version, intra_type)
    end = time.time()
    return {
        'Filename': filename,
        'Version': version,
        'Intra_Type': intra_type,
        'Path': path,
        'Initial_score': initial_score,
        'Score': score,
        'Time': (end - start)
    }


if __name__ == "__main__":
    test_start = time.time()
    for filename in ['TSPD.csv', 'TSPA.csv', 'TSPB.csv', 'TSPC.csv']: 
        df = pd.read_csv(filename, header=None, sep=';', names=['x', 'y', 'cost'])
        costs = df['cost'].values
        df.shape[0]
        distance_matrix = create_distance_matrix(df)
        mapping_out_files= {
            'TSPA.csv' : 'Random_TSPA_out.csv',
            'TSPB.csv' : 'Random_TSPB_out.csv',
            'TSPC.csv' : 'Random_TSPC_out.csv',
            'TSPD.csv' : 'Random_TSPD_out.csv'
        }
        mapping_in_files= {
            'TSPA.csv' : 'A_results.csv',
            'TSPB.csv' : 'B_results.csv',
            'TSPC.csv' : 'C_results.csv',
            'TSPD.csv' : 'D_results.csv',
        }
        initial = pd.read_csv(mapping_in_files[filename], header=None, sep=',')
        #result = initial.apply(lambda row: row.tolist(), axis=1)
        result = initial.iloc[:, 0:100].values.tolist()
        # print(np.array(result[0]))
        results = Parallel(n_jobs=-1)(delayed(evaluate_solution)(filename, np.array(result[i]), version, intra_type, costs, distance_matrix)
                                    for version in ['steepest', 'greedy']
                                    for intra_type in ['nodes', 'edges']
                                    for i in range(200)
                                    )

        results_df = pd.DataFrame(results)
        results_df.to_csv(mapping_out_files[filename])
        test_end = time.time()
        print(test_end - test_start)
