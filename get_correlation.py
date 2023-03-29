import os
import pandas as pd
import argparse
from omegaconf import OmegaConf
import numpy as np
from scipy import stats

##########

def main(file1_path, file2_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path).to_numpy()
    
    df1 = df1.to_numpy()
    
    df1 = df1[np.triu_indices(df1.shape[0], k = 1)]
    df2 = df2[np.triu_indices(df2.shape[0], k = 1)]
    
    distance = stats.spearmanr(df1, df2).correlation
    
    print('Spearman correlation', distance)
    
    # dump_to_json(distances, path, name=file_name)


if __name__ == "__main__":
    file1_path = '/home/harishbabu/projects/ProtoTree/runs/010-cub_190_imgnet_224-dth=9-ep=100/pruned_and_projected/prototree_species_distances_phylosorted.csv'
    file2_path = '/home/harishbabu/projects/ProtoTree/analysis/phylo_trees/phylogenetic_species_distances_cub_phylosorted.csv'
    main(file1_path, file2_path)