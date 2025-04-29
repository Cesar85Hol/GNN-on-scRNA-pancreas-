import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def read_rawdata(path):
    df=pd.read_csv(path,sep=' ')

    print("Shape check [#genes x # cells] :", df.shape)  
    return df

#FILTERING FUNCTIONS
def filter_genes_expressed(df,minGenes):
    # Calculation of number of genes expressed per cell
    genes_per_cell = (df > 0).sum(axis=0)
    # Select cells expressing at least > minGenes
    valid_cells = genes_per_cell[genes_per_cell >= minGenes].index
    df_filtered = df.loc[:, valid_cells]
    print("Cells remained after filtering:", df_filtered.shape[1])
    return df_filtered

def filter_genes_not_informative(df,minCells):
    # Cells expressed per gene
    cells_per_gene = (df > 0).sum(axis=1)
    # Keep "minCells" Cells
    valid_genes = cells_per_gene[cells_per_gene >= minCells].index
    df_filtered = df.loc[valid_genes, :]
    print("Genes remained after filtering:", df.shape[0])
    return df_filtered

#PREPROCESS FUNCTIONS

def normalize(df,scaleFactor):
    tots=df.sum(axis=0)
    df_norm=df.divide(tots,axis=1)*scaleFactor
    df_norm=np.log1p(df_norm) # loagrithm natural of (1+x) to manage 0
    
    print(np.round(df_norm.sum(axis=0).describe(), 2))
    return df_norm

def reduce_dimentionality(df,dim=50): #tip you can use the more informative genes instead of PCA
    X=df.T.values # transpose fro PCA pruposes --> (cells x genes)
    pca=PCA(n_components=dim)
    X_pca=pca.fit_transform(X)
    print("New dimensions: " ,X_pca.shape)
    return X_pca

