import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx,to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def adjacent_matrix(array,threshold=0.3,top_n=10):
    
    #Calculate 3 different similarity matrices 
    # 1. Pearson correlation
    pearson_sim = np.corrcoef(array) #row x row --> cells per cells
    thresh_pearson = threshold      #best 0.6
    print("Pearson correlation matrix")
    adj_pearson = adjacency_from_similarity(pearson_sim, thresh_pearson)
    
    # 2. Cosine similarity
    #tensor = torch.tensor(array, dtype=torch.float)
    #cosine_sim = torch.nn.functional.cosine_similarity(tensor[0].unsqueeze(0), tensor, dim=1)
    cosine_sim = cosine_similarity(array)
    thresh_cosine = threshold       #best 0.9
    print("Cosine similarity matrix")
    adj_cosine = adjacency_from_similarity(cosine_sim, thresh_cosine)
    
    # 3. Euclidean distance
    euclidean_dist = squareform(pdist(array, metric='euclidean'))
    euclidean_sim = 1 / (1 + euclidean_dist)  # Similarità inversa alla distanza
    thresh_euclidean = threshold    #best 0.25
    print("Euclidean distance matrix")
    adj_euclidean = adjacency_from_similarity(euclidean_sim, thresh_euclidean)
    
    return adj_pearson, adj_cosine, adj_euclidean

def adjacency_from_similarity(sim_matrix, threshold):
    adj = (sim_matrix >= threshold).astype(int)
    np.fill_diagonal(adj, 0)  # niente self-loop
    # Count #edges on uppert part
    edges_count = np.triu(adj).sum()
    print(f"Threshold {threshold}: around {edges_count} graph connections")
    return adj

def get_top_n_correlation_cells(corr_matrix, n_cells,top_n=10):
    # Build edges based on top-10 correlations for each cell
    
    edges_corr = set()
    for i in range(n_cells):
        # get correlation values for cell i
        corr_i = corr_matrix[i].copy()
        corr_i[i] = -np.inf  # exclude self by setting self-correlation to -inf
        # find indices of top k highest correlations
        topk_idx = np.argsort(corr_i)[-top_n:]
        for j in topk_idx:
            # add edge i->j
            edges_corr.add((i, j))
            # also add reverse edge j->i to make it undirected
            edges_corr.add((j, i))

    # Convert edge list to edge_index format (2 x E array)
    edge_index_corr = np.array(list(edges_corr), dtype=int).T
    print("Correlation graph edges:", edge_index_corr.shape[1])


def createGraph(adj_matrix,X_pca, top_n=0):
    # Obtain (i,j) where adj_matrix is 1
    edge_indices = np.array(np.nonzero(adj_matrix))
    # edge_indices is an array 2 x E (2 rows: lista i and list of j)
    # Convert to pytorch tensors
    edge_index = torch.tensor(edge_indices, dtype=torch.long)

    # Create tensor of feature nodes
    x = torch.tensor(X_pca, dtype=torch.float)

    # Creiamo l'oggetto Data di PyG
    data = Data(x=x, edge_index=edge_index)
    print(data)
    if top_n > 0:
        
        #creating a subgraph to better visualize the graph
        subG=extractsubGraph(adj_matrix, top_n=top_n)
        plt.figure(figsize=(10, 10))
        nx.draw_spring(subG, node_size=10, with_labels=False, edge_color='gray', alpha=0.7)
        plt.title(f"SubGraph - {top_n} nodes with higher degree")
        plt.show()
    else:
        G=to_networkx(data, to_undirected=True)
        #Visualize the graph
        plt.figure(figsize=(10, 10))
        pos=nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=False,
                node_size=10, node_color='skyblue', 
                alpha=0.5)
        plt.title("Entire graph PCA 50 components")
        plt.show()
    return data,edge_index

def extractsubGraph(adj_matrix, top_n=10):
    # Extract the top_n nodes with the highest degree
    G = nx.from_numpy_array(adj_matrix)
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_n]
    
    # Create a subgraph with the top_n nodes
    subgraph = G.subgraph(sorted_nodes)
    print(subgraph)
    # Convert to PyTorch Geometric Data object
    data = from_networkx(subgraph)
    
    return subgraph
    


