import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Tuple
import copy
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Tuple
import copy
import time
import numpy as np
from joblib import Parallel, delayed
from itertools import combinations, product
from queue import PriorityQueue, Empty
import math

MAX_DIST = 2147483647

best_paths = {'A': [31, 111, 14, 80, 95, 169, 8, 26, 92, 48, 106, 160, 11, 152, 130, 119, 109, 189, 75, 1, 177, 41, 137, 174, 199, 150, 192, 175, 114, 4, 77, 43, 121, 91, 50, 149, 0, 19, 178, 164, 159, 143, 59, 147, 116, 27, 96, 185, 64, 20, 71, 61, 163, 74, 113, 195, 53, 62, 32, 180, 81, 154, 102, 144, 141, 87, 79, 194, 21, 171, 108, 15, 117, 22, 55, 36, 132, 128, 76, 161, 153, 88, 127, 186, 45, 167, 101, 99, 135, 51, 112, 66, 6, 156, 98, 190, 72, 94, 12, 73], 'B':  [122, 143, 179, 197, 183, 34, 31, 101, 38, 103, 131, 121, 127, 24, 50, 112, 154, 134, 25, 36, 165, 37, 137, 88, 55, 4, 153, 80, 157, 145, 136, 61, 73, 185, 132, 52, 12, 107, 189, 170, 181, 147, 159, 64, 129, 89, 58, 72, 114, 85, 166, 59, 119, 193, 71, 44, 196, 117, 150, 162, 158, 67, 3, 156, 91, 70, 51, 174, 188, 140, 148, 141, 130, 142, 53, 69, 115, 82, 63, 8, 16, 18, 29, 33, 19, 190, 198, 135, 95, 172, 182, 2, 5, 128, 66, 169, 0, 57, 99, 92], 'C': [22, 195, 55, 36, 132, 128, 145, 76, 161, 153, 88, 127, 186, 45, 167, 101, 99, 135, 51, 5, 112, 66, 6, 172, 156, 98, 190, 72, 94, 12, 73, 31, 95, 169, 8, 26, 92, 48, 11, 152, 130, 119, 109, 189, 75, 1, 177, 41, 137, 199, 192, 43, 77, 4, 114, 91, 121, 50, 149, 0, 69, 19, 178, 164, 34, 159, 143, 59, 147, 116, 27, 96, 185, 64, 20, 71, 61, 113, 74, 163, 155, 93, 62, 32, 180, 81, 154, 102, 144, 141, 87, 79, 194, 21, 157, 171, 108, 15, 117, 53], 'D': [165, 37, 137, 99, 57, 0, 169, 66, 26, 92, 122, 143, 179, 127, 24, 121, 131, 103, 38, 101, 31, 197, 183, 34, 5, 2, 182, 172, 95, 135, 198, 190, 19, 33, 29, 18, 16, 8, 63, 82, 115, 69, 113, 32, 53, 142, 130, 141, 148, 140, 188, 174, 51, 70, 91, 156, 3, 45, 67, 114, 72, 58, 89, 159, 147, 64, 129, 85, 166, 162, 150, 117, 196, 44, 71, 59, 119, 193, 139, 97, 107, 12, 52, 132, 185, 73, 61, 136, 79, 145, 157, 80, 153, 4, 55, 88, 36, 25, 134, 154]
}


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
    copy_path = np.array(path)
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

def get_plot_values(nodes : Dict[int, Tuple[int, int, int]], solution: List[int], costs: List[int]) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    x_coords = [nodes[node][0] for node in list(nodes.keys())]
    y_coords = [nodes[node][1] for node in list(nodes.keys())]
    solution = solution + [solution[0]]
    path_x_coords = [nodes[node][0] for node in solution]
    path_y_coords = [nodes[node][1] for node in solution]
    new_costs = [(cost/max(costs))*100 for cost in costs]
    min_cost, max_cost = min(costs), max(costs)
    power = 2
    normalized_costs = [((cost - min_cost) / (max_cost - min_cost)) ** power for cost in costs]
    colors = plt.cm.RdBu(normalized_costs)
    return x_coords, y_coords, new_costs, path_x_coords, path_y_coords, colors

def plot_path(path, nodes, costs):
    x_coords, y_coords, new_costs, path_x_coords, path_y_coords, colors = get_plot_values(nodes, path, costs)
    plt.figure(figsize=(10,8))
    plt.scatter(x_coords, y_coords, color=colors, marker='o', s=new_costs, label='Cities')
    plt.plot(path_x_coords, path_y_coords, linestyle='-', marker='o', markersize=0, color='blue', label='Path', alpha = 0.7)
    plt.show()

def read_data(filename='TSPA.csv', PATH='./'):
    df = pd.read_csv(PATH + filename, names=["x", "y", "costs"], sep=';', header=None)
    nodes = {}
    for idx, row in enumerate(df.values):
        x, y, cost = map(int, row)
        nodes[idx] = (x, y, cost)
    D = create_distance_matrix(df)
    return nodes, df['costs'].values, D


class Delta:
  def __init__(self, delta_value: int, delta_type: str, values: tuple):
      self.values = values
      self.delta_type = delta_type
      self.delta_value = delta_value
  def create_path(self, path):
      if self.delta_type == 'inter':
          index_of_old_node, new_value = self.values
          modified_path = copy.deepcopy(path)
          modified_path[index_of_old_node] = new_value
          return modified_path
      elif self.delta_type == 'edges_intra':
          n1, n2 = self.values
          if n1 < n2:
              modified_path = np.concatenate((path[:n1], [path[n1]], [path[n2]], path[n1+1:n2][::-1],  path[n2+1:]))
          else:
              modified_path = np.concatenate((path[n2+1:n1], [path[n1]], [path[n2]], path[:n2][::-1], path[n1+1:][::-1]))
          return modified_path


def inter(path, distance_matrix, costs):
    if any(isinstance(item, float) for item in path):
        print("edges inter start")
    og_nodes = set(path)
    nodes = set(list(range(distance_matrix.shape[0])))
    changes = product(og_nodes, nodes - og_nodes)
    changes = list(changes)
    random.shuffle(changes)
    node_indices = {node: index for index, node in enumerate(path)}
    deltas = []
    for change in changes:
        old_node, new_node = change
        delta_value = 0
        index_of_old_node = node_indices[old_node]
        prev = path[(index_of_old_node - 1) % len(path)]
        next_node = path[(index_of_old_node + 1) % len(path)]
        old_dist = distance_matrix[prev, old_node] \
                  + distance_matrix[old_node, next_node]
        new_dist = distance_matrix[prev, new_node] \
                  + distance_matrix[new_node, next_node]
        delta_value += new_dist - old_dist  # lower is better
        delta_value += costs[new_node] - costs[old_node]  # lower is better
        delta = Delta(delta_value, 'inter', (index_of_old_node, new_node))
        if delta_value < 0:
            deltas.append(delta)
    return deltas

def edges_intra(path, distance_matrix):
    if any(isinstance(item, float) for item in path):
        print("edges_intra start")
    pairs1 = np.column_stack((path[:-1], path[1:]))
    cyclic_pair = np.array([path[-1], path[0]])
    all_edges = np.vstack((pairs1, cyclic_pair))
    edge_swaps = combinations(all_edges, 2)
    edge_swaps = list(edge_swaps)
    random.shuffle(edge_swaps)
    node_indices = {node: index for index, node in enumerate(path)}
    deltas = []
    for swap in edge_swaps:
        edge1, edge2 = swap
        if edge1[1] == edge2[0] or edge1[0] == edge2[1]:
            continue
        delta_value = (
            distance_matrix[edge1[0], edge2[0]]
            + distance_matrix[edge1[1], edge2[1]]
        ) - (
            distance_matrix[edge1[0], edge1[1]]
            + distance_matrix[edge2[1], edge2[0]]
        )
        index1 = node_indices[edge1[0]]
        index2 = node_indices[edge2[0]]
        delta = Delta(delta_value, 'edges_intra', (index1, index2))
        if delta_value < 0:
            deltas.append(delta)
    return deltas

def local_search(current_path, costs, distance_matrix):
    current_path = np.array(current_path)
    while True:
        neighbourhood_inter_deltas = inter(current_path, distance_matrix, costs)
        neighbourhood_intra_deltas = edges_intra(current_path, distance_matrix)
        neighbourhood_deltas = np.array(neighbourhood_inter_deltas + neighbourhood_intra_deltas)
        if neighbourhood_deltas.size > 0:
            min_idx = np.argmin([delta.delta_value for delta in neighbourhood_deltas])
            best_delta = neighbourhood_deltas[min_idx]
            # if any(isinstance(item, float) for item in current_path):
                # print("before path creation in local search")
            current_path = best_delta.create_path(current_path)
            # if any(isinstance(item, float) for item in current_path):
                # print("after path creation in local search")
        else:
            break
    return list(current_path)