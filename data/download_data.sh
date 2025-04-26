#!/bin/bash

# URL 
URL1="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186469/suppl/GSE186469_ctrl.data.txt.gz"
URL2="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186469/suppl/GSE186469_t2d.data.txt.gz"
# Destination path
DEST="../data/raw/ctrl_data.txt.gz"
DEST2="../data/raw/t2d_data.txt.gz"


# Scarica il file
wget -O "$DEST" "$URL1"
wget -O "$DEST2" "$URL2"

#!wget -O ctrl_data.txt.gz "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186469/suppl/GSE186469_ctrl.data.txt.gz"
#!wget -O t2d_data.txt.gz "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186469/suppl/GSE186469_t2d.data.txt.gz"
