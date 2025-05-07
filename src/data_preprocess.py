import pandas as pd
import numpy as np
from scipy.io import mmread
from scipy import sparse
import os
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
def read_rawdata(path,type=0):
    '''
    @type: 0 for raw, 1 for preprocessed
    '''
    if type ==0:
        if path.endswith('.gz'):
            # Read the matrix market file
            counts_coo = mmread(path)
            # Convert to a dense DataFrame
            df = counts_coo.tocsc()
            print("Shape check [#genes x # cells] :", df.shape) 
        elif path.endswith('.txt'):
            df= pd.read_csv(path, header=None)
            print("Shape txt file :", df.shape)
        elif path.endswith('.tsv'):
            df= pd.read_csv(path, sep='\t', header=0)
            print("Shape tsv file :", df.shape)

    else:
        df=pd.read_csv(path,sep=' ')
        print("Shape check [#genes x # cells] :", df.shape)
    
    return df

#FILTERING FUNCTIONS
def filter_genes_expressed(df,labels,cell_meta,minGenes):

    '''Work with dataframe
    # Calculation of number of genes expressed per cell
    genes_per_cell = (df > 0).sum(axis=0)
    # Select cells expressing at least > minGenes
    valid_cells = genes_per_cell[genes_per_cell >= minGenes].index
    df_filtered = df.loc[:, valid_cells]

    #same with arrays
    genes_per_cell = np.sum(data > 0, axis=0)
    valid_cells_mask = genes_per_cell >= minGenes
    df_filtered = df[:, valid_cells_mask]
    '''
    # Work with ndarray

    # Calculation of number of genes expressed per cell
    genes_per_cell = np.diff(df.indptr) #
    # Select cells expressing at least > minGenes
    low_quality_mask = genes_per_cell < minGenes
    print("Cells with <500 expressed genes: ", low_quality_mask.sum())
    
    df_filtered = df[:, ~low_quality_mask]
    labels_filtered = labels[~low_quality_mask] 
    print("Cells remained after filtering:", df_filtered.shape[1])
    # Filter cell metadata dataframe
    cell_meta = cell_meta.loc[~low_quality_mask].reset_index(drop=True)
    
    return df_filtered,labels_filtered,cell_meta

def filter_genes_not_informative(data,gene_names,minCells):
    '''
    # Cells expressed per gene
    cells_per_gene = (df > 0).sum(axis=1)
    # Keep "minCells" Cells
    valid_genes = cells_per_gene[cells_per_gene >= minCells].index
    df_filtered = df.loc[valid_genes, :]
    print("Genes remained after filtering:", df_filtered.shape[0])
    '''
    data_csr = data.tocsr()
    cells_per_gene = np.diff(data_csr.indptr)  # non-zero per riga
    gene_mask = cells_per_gene >= minCells             # True per geni da mantenere

    print(f"Genes with < {minCells} expressed cells:", (~gene_mask).sum())
    data_filtered = data_csr[gene_mask, :]
    gene_names_filtered =gene_names[gene_mask]# [g for g, drop in zip(gene_names, gene_mask) if not drop]

    print("Filtered dimensions:", data_filtered.shape)
    return data_filtered, gene_names_filtered


#PREPROCESS FUNCTIONS

def normalize(df,scaleFactor):
    tots=df.sum(axis=0)
    df_norm=df.divide(tots,axis=1)*scaleFactor
    df_norm=np.log1p(df_norm) # loagrithm natural of (1+x) to manage 0
    
    print(np.round(df_norm.sum(axis=0).describe(), 2))
    return df_norm

def log_normalize(sparse_matrix):
    #Seurat procedure: scale to 10000 -> log1p
    col_sum = np.array(sparse_matrix.sum(axis=0)).ravel()
    scale_factors = 1e4 / col_sum
    D = sparse.diags(scale_factors)
    matrix_norm = sparse_matrix.dot(D)
    col_sum_norm = np.array(matrix_norm.sum(axis=0)).ravel()
    print(col_sum_norm[:5])
    matrix_norm.data = np.log1p(matrix_norm.data)
    return matrix_norm


def reduce_dimentionality(df,dim=50,method='pca'): #tip you can use the more informative genes instead of PCA
    if method=='pca':
        X=df.T.values # transpose fro PCA pruposes --> (cells x genes)
        pca=PCA(n_components=dim)
        X_pca=pca.fit_transform(X)
        print("New dimensions: " ,X_pca.shape)
        return X_pca
    elif method=='svd':
        svd = TruncatedSVD(n_components=dim,random_state=0)
        X_svd = svd.fit_transform(df.transpose())
        print("New dimensions: " ,X_svd.shape)
        print("Variance explained on first 5 PC:", svd.explained_variance_ratio_[:5])
        return X_svd   

def select_highly_variable_genes(df, genes, n_top_genes=2000):
    # Calculate the mean and variance of each gene
    X=df.toarray() 
    gene_mean = np.mean(X, axis=1)
    gene_var = np.var(X, axis=1)

    # Calculate the dispersion (variance / mean)
    gene_dispersion = gene_var / gene_mean

    # Ignora eventuali NaN (geni con mean=0, già filtrati comunque)
    dispersion = np.nan_to_num(gene_dispersion, nan=0.0)
    # Seleziona indice dei top 1000 geni per dispersione
    top_hvg_indices = np.argsort(dispersion)[-n_top_genes:]
    # Filtra la matrice originale mantenendo solo questi geni
    data_hvg = df[top_hvg_indices, :]
    genes_hvg= [genes.iloc[i] for i in top_hvg_indices]
    #genes_hvg = genes.iloc[top_hvg_indices]
    print("Dimensions matrix HVG:", data_hvg.shape)
    hvg_vars = gene_var[top_hvg_indices]
    print("mean variance of HVG selected:", hvg_vars.mean(),
        " - mean variance other genes:", gene_var[np.argsort(dispersion)[:-1000]].mean())

    return data_hvg,genes_hvg,top_hvg_indices
