import networkx as nx
import matplotlib.pyplot as plt


def show_nx_graph(G):
  pos = nx.spring_layout(G)
  nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, alpha=0.8)
  plt.show()


def get_total_paths_from_adjacency_matrix(matrix):
  G = transform_dag_adjacency_matrix2graph(matrix)
  return get_total_paths(G, 0, matrix.shape[0]-1)


def get_longest_path_from_adjacency_matrix(matrix):
  G = transform_dag_adjacency_matrix2graph(matrix)
  return nx.dag_longest_path_length(G)


def get_total_paths(G, source, target):
  return len(list(nx.all_simple_paths(G, source, target)))


def transform_dag_adjacency_matrix2graph(matrix):
  G = nx.DiGraph()
  i_max, j_max = matrix.shape
  for i in range(i_max):
    for j in range(j_max):
      if matrix[i, j] == 1:
        G.add_edge(i, j)
  return G
