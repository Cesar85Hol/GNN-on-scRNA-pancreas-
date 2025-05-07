# GNN-on-scRNA-pancreas-
Example of application of GNN on real dataset : human pancreas.

# Requirements
- Pytorch
- Pytorch geometric

# Dataset
- E-MTAB-506. Segerstolpe et al. (https://www.ebi.ac.uk/gxa/sc/experiments/E-MTAB-5061/results/cell-plots?ref=biostudies&plotOption=5&colourBy=inferred+cell+type+-+authors+labels) 
- Downloaded from https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments/E-MTAB-5061/ 
- pancreatic islet cells from 10 donors (6 healthy, 4 with T2D)
- 3386 cells of scRNA (raw data)

**Workflow**:
1. Preprocess data
- Filtering: expressed & not informative genes  
- Normalization 
- Feature selection: PCA & Highly variable genes 
- Build graph  
2. GNN - cluster preprocess data
3. GNN - classifier to find important cells in T2D patients (diabetes type 2)
- 
# Project structure
```bash
project-root/
├── data/
│   ├── raw/                # Dati originali scaricati
│   └── processed/          # Dati dopo il pre-processing
├── notebooks/              # Notebook Jupyter per esplorazione e analisi
├── src/                    # Codice sorgente
│   ├── data_preprocessing/ # Script per pre-processing dei dati
│   ├── graph_construction/ # Script per costruzione del grafo
│   ├── GNN_classificators/ # Script per implementazione classificatori GNN
│   ├── GNN_clustering/     # Script per implementazione clustering
│   └── models/             # Definizione e addestramento dei modelli GNN
├── results/                # Risultati dell'analisi (grafici, metriche, ecc.)
├── requirements.txt        # Dipendenze del progetto
├── README.md               # Descrizione del progetto
└── .gitignore              # File e cartelle da ignorare in Git
```
# How to use
- Start with the notebook main_New in folder notebooks
