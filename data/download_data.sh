#!/bin/bash

# URL main
#URL1="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186469/suppl/GSE186469_ctrl.data.txt.gz"
#URL2="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186469/suppl/GSE186469_t2d.data.txt.gz"
# Destination path
#DEST="../data/raw/ctrl_data.txt.gz"
#DEST2="../data/raw/t2d_data.txt.gz"

# URL ArrayExpress E-MTAB-5061
URL1="https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments/E-MTAB-5061/E-MTAB-5061.aggregated_counts.mtx.gz"
URL2="https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments/E-MTAB-5061/E-MTAB-5061.aggregated_counts.mtx_rows"
URL3="https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments/E-MTAB-5061/E-MTAB-5061.aggregated_counts.mtx_cols"

DEST="../data/raw/data.mtx.gz"
DEST2="../data/raw/data_rows.txt"
DEST3="../data/raw/data_cols.txt"





# Download data
wget -O "$DEST" "$URL1"
wget -O "$DEST2" "$URL2"
wget -O "$DEST3" "$URL3"
