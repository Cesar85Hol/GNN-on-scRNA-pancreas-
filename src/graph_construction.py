import numpy as np
import torch
from torch_geometric.data import Data

def adjacent_matrix(array,threshold=0.3):
    # Calcoliamo la matrice di correlazione Pearson tra cellule
    # X_pca ha dimensioni (n_cellule, n_componenti)
    # threshold is % of correlation
    corr_matrix = np.corrcoef(array, rowvar=False)  

    print("Dimension of matrix correlation:", corr_matrix.shape)
    # Crate adjacent matrix
    adj_matrix = (corr_matrix >= threshold).astype(int)
    # Remove autocorrelation on diagonal
    np.fill_diagonal(adj_matrix, 0)
    # Count #edges on uppert part
    edges_count = np.triu(adj_matrix).sum()
    print(f"Threshold {threshold}: around {edges_count} graph connections")
    return adj_matrix

def createGraph(adj_matrix,X_pca):
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
    return data