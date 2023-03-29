from prototree.prototree import ProtoTree
from PIL import Image
import glob
import os
import torch
from util.data import get_dataloaders, get_data
from util.args import load_args
import subprocess
import numpy as np
import pandas as pd
import copy
import argparse
from subprocess import check_call
from PIL import Image
import torch
import math
from prototree.prototree import ProtoTree
from prototree.branch import Branch
from prototree.leaf import Leaf
from prototree.node import Node
from util.visualize import _gen_dot_edges, _gen_dot_nodes
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# from util.visualize_prediction import upsample_local
from util.visualize_prediction_temp import upsample_local
import collections

import seaborn as sns

from ete3 import Tree

# from prototree import ProtoTree
args = load_args('/home/harishbabu/projects/ProtoTree/runs/010-cub_190_imgnet_224-dth=9-ep=100/metadata')

# node_image_path = os.path.join(log_dir, 'pruned_and_projected/node_vis')

log_dir = args.log_dir
trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args)
tree = ProtoTree.load(os.path.join(log_dir, 'checkpoints/pruned_and_projected'))
tree = tree.eval()
# breakpoint()

def get_common_path_score(tree, classes):

    class_to_leaves = {classname: [] for classname in classes}
    idx_to_class = {i: classname for i, classname in enumerate(classes)}

    for leaf in tree.leaves:
        class_idx = torch.argmax(leaf.distribution()).item()
        class_to_leaves[idx_to_class[class_idx]].append(leaf)

    for i in range(len(classes)):
        class1_leaves = class_to_leaves[i]
        for j in range(i, len(classes)):
            class2_leaves = class_to_leaves[j]
            scores_list = []
            for class1_leaf in class1_leaves:
                for class2_leaf in class2_leaves:
                    common_parent = tree.get_common_parent(class1_leaf, class2_leaf)
                    common_path = tree.path_to(common_parent)
                    current_node = common_parent
                    score = 0
                    while current_node is not None:
                        if tree._parents[current_node].r == current_node:
                            score += 1
                        else:
                            score -= 1
                        current_node = tree._parents[current_node]

                    scores_list.append(score)

    return score

# def visualize_each_path(args, tree, dataloader, classes):
#     log_dir = args.log_dir
#     trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args)
#     tree = ProtoTree.load(os.path.join(log_dir, 'checkpoints/latest'))


def visualize_each_path(tree: ProtoTree, folder_name: str, args: argparse.Namespace, classes:tuple):
    destination_folder=os.path.join(args.log_dir,folder_name)
    upsample_dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images), folder_name)
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)
    if not os.path.isdir(destination_folder + '/node_vis'):
        os.mkdir(destination_folder + '/node_vis')

    all_paths = []
    n_cols = 0
    for k, leaf in enumerate(tree.leaves):
        path = tree.path_to(leaf)
        path_copy = []
        for leaf in path:
            path_copy.append(copy.deepcopy(leaf))

        for i in range(len(path_copy)):
            if isinstance(path_copy[i], Branch):
                if path[i].l == path[i+1]:
                    path_copy[i].r = None
                    path_copy[i].l = path_copy[i+1]
                else:
                    path_copy[i].l = None
                    path_copy[i].r = path_copy[i+1]

        all_paths.append(path_copy)
        n_cols = max(n_cols, len(path_copy))

    n_rows = len(all_paths)    


    with torch.no_grad():
        s = 'digraph T {rankdir=LR;rank=same;margin=0;ranksep=".03";nodesep="0.05";splines="false";\n'
        for i, path in enumerate(all_paths):
            s += 'node [shape=rect, label=""];\n'
            s += _gen_dot_nodes(path[0], destination_folder, upsample_dir, classes, node_suffix='_'+str(i))
            s += _gen_dot_edges(path[0], classes, node_suffix='_'+str(i))[0]
            s += '\n'
        s += '}\n'

    with open(os.path.join(destination_folder,'paths.dot'), 'w') as f:
        f.write(s)
   
    from_p = os.path.join(destination_folder,'paths.dot')
    to_pdf = os.path.join(destination_folder,'paths.pdf')
    # check_call('gvpack -u %s | dot -Tpdf -o %s'%(from_p, to_pdf), shell=True)
    check_call('dot -Tpdf -Gmargin=0 %s -o %s'%(from_p, to_pdf), shell=True)


def copy_decision_path(path):
    path_copy = []
    for node in path:
        path_copy.append(copy.deepcopy(node))

    for i in range(len(path_copy)):
        if isinstance(path_copy[i], Branch):
            if path[i].l == path[i+1]:
                path_copy[i].r = None
                path_copy[i].l = path_copy[i+1]
            else:
                path_copy[i].l = None
                path_copy[i].r = path_copy[i+1]
    return path_copy


def visualize_each_path_with_input(tree: ProtoTree, folder_name: str, args: argparse.Namespace, classes:tuple):
    # trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args)
    trainset, projectset, testset, classes, shape = get_data(args)
    dataset = testset
    cuda = not args.disable_cuda and torch.cuda.is_available()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=cuda
                                             )
    
    leaves = list(tree.leaves)

    class_to_leaf = collections.defaultdict(list)
    leaf_dict = collections.defaultdict(lambda: [0,None,None,'','']) # max_path_prob, image_tensor, path(list of nodes), image_path, path_of_image_with_bounding_box

    for leaf in leaves:
        class_to_leaf[torch.argmax(leaf.distribution()).item()].append(leaf)

    # for idx, (image, label) in enumerate(dataset):
    #     image = image.unsqueeze(0).cuda()
    #     _, info = tree.forward(image)
    #     for leaf in class_to_leaf[label]:
    #         if info['pa_tensor'][leaf.index] > leaf_dict[leaf.index][0]:
    #             leaf_dict[leaf.index][0] = info['pa_tensor'][leaf.index]
    #             leaf_dict[leaf.index][1] = image
    #             leaf_dict[leaf.index][3] = dataset.imgs[idx][0]

    for idx, (image, label) in enumerate(dataloader):
        image = image.cuda()
        _, info = tree.forward(image)
        # breakpoint()
        for i in range(image.shape[0]):
            for leaf in class_to_leaf[label[i].item()]:
                if info['pa_tensor'][leaf.index][i].item() > leaf_dict[leaf.index][0]:
                    leaf_dict[leaf.index][0] = info['pa_tensor'][leaf.index][i].item()
                    leaf_dict[leaf.index][1] = image[i].unsqueeze(0)
                    leaf_dict[leaf.index][3] = dataset.imgs[idx][0]

    destination_folder=os.path.join(args.log_dir,folder_name)
    upsample_dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images), folder_name)
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)
    if not os.path.isdir(destination_folder + '/node_vis'):
        os.mkdir(destination_folder + '/node_vis')

    # all_paths = []
    # n_cols = 0
    # for k, leaf in enumerate(tree.leaves):
    #     path = tree.path_to(leaf)
    #     path_copy = []
    #     for node in path:
    #         path_copy.append(copy.deepcopy(node))

    #     for i in range(len(path_copy)):
    #         if isinstance(path_copy[i], Branch):
    #             if path[i].l == path[i+1]:
    #                 path_copy[i].r = None
    #                 path_copy[i].l = path_copy[i+1]
    #             else:
    #                 path_copy[i].l = None
    #                 path_copy[i].r = path_copy[i+1]

    #     # all_paths.append(path_copy)
    #     leaf_dict[leaf.index][2] = path_copy

    for k, leaf in enumerate(tree.leaves):
        path = tree.path_to(leaf)
        leaf_dict[leaf.index][2] = path

    for leaf_index in leaf_dict:
        image = leaf_dict[leaf_index][1]
        decision_path = leaf_dict[leaf_index][2]
        sample_dir = leaf_dict[leaf_index][3]
        leaf_dict[leaf.index][4] = upsample_local(tree, image, sample_dir, decision_path, args)

    with torch.no_grad():
       
        s = 'digraph T {rankdir=LR;rank=same;margin=0;ranksep=".03";nodesep="0.05";splines="false";\n'
        for i, leaf_index in enumerate(leaf_dict):
            path = copy_decision_path(leaf_dict[leaf.index][2])
            node_suffix=''+str(i)
            s += 'node [shape=rect, label=""];\n'

            s += _gen_dot_nodes(path[0], destination_folder, upsample_dir, classes, node_suffix)
            
            filename_format = "{}_bounding_box_nearest_patch_of_image.png"
            bb_image_dir = leaf_dict[leaf.index][4]
            for node in path[:-1]:
                s += '{}[image="{}" xlabel="{}" fontsize=6 labelfontcolor=gray50 fontname=Helvetica];\n'.format(str(node.index)+(node_suffix*2),
                                                                                                                os.path.join(bb_image_dir, filename_format.format(0)), # replace 0 with node.index
                                                                                                                node.index)
                
            
            # s = '{}[imagepos="tc" imagescale=height image="{}" label="{}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'.format(str(path[-1].index)+node_suffix, filename, str_targets)

            s += _gen_dot_edges(path[0], classes, node_suffix)[0]
            for j, node in enumerate(path[:-1]):
                if path[j].r == path[j+1]:
                    label = 'Present'
                else:
                    label = 'Absent'
                s += '{}->{} [label="{}" fontsize=10 tailport="s" headport="n" fontname=Helvetica];\n'.format(str(node.index)+node_suffix,
                                                                                              str(node.index)+(node_suffix*2),
                                                                                              label)
                # "{" + "rank = same; {}; {}".format(str(node.index)+node_suffix,
                #                                     str(node.index)+(node_suffix*2)) + "};"
            s += '\n'
        s += '}\n'

    with open(os.path.join(destination_folder,'paths_with_sample.dot'), 'w') as f:
        f.write(s)
   
    from_p = os.path.join(destination_folder,'paths_with_sample.dot')
    to_pdf = os.path.join(destination_folder,'paths_with_sample.pdf')
    check_call('dot -Tpdf -Gmargin=0 %s -o %s'%(from_p, to_pdf), shell=True)


def distance_between_species(tree, folder_name, args, classes):
    destination_folder=os.path.join(args.log_dir,folder_name)
    # decision_vector = collections.defaultdict(np.zeros((tree.nodes, 1)))
    decision_vector = {}
    nodes_count = 2**(args.depth+1)-1
    for i in range(len(classes)):
        decision_vector[i] = np.zeros((nodes_count, 1))
    # class_to_leaf = collections.defaultdict(list)
    class_to_branches = collections.defaultdict(set)

    for leaf in tree.leaves:
        # class_to_leaf[torch.argmax(leaf.distribution()).item()].append(leaf)
        for node in tree.path_to(leaf)[:-1]: # excluding leaf node
            class_to_branches[torch.argmax(leaf.distribution()).item()].add(node)

    for idx in range(len(classes)):
        for node in class_to_branches[idx]:
            if (node.r in class_to_branches[idx]) and (node.l in class_to_branches[idx]):
                decision_vector[idx][node.index] = 0
            elif node.r in class_to_branches[idx]:
                decision_vector[idx][node.index] = 1
            elif node.l in class_to_branches[idx]:
                decision_vector[idx][node.index] = -1

    heatmap = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
            heatmap[i][j] = np.linalg.norm(decision_vector[i] - decision_vector[j])
            heatmap[j][i] = heatmap[i][j]

    plt.imsave(os.path.join(destination_folder, 'prototree_correlation.png'), heatmap)


def cosine_distance(vector1, vector2):
    return np.dot(vector1.T, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))


def singlelink_distance_between_vectors(vector_set1, vector_set2):
    min_distance = np.iinfo(np.int64).max
    closest_vectors = (None, None)
    for vector1 in vector_set1:
        for vector2 in vector_set2:
            dist = cosine_distance(vector1, vector2)
            if dist < min_distance:
                min_distance = dist
                closest_vectors = (vector1, vector2)

    return min_distance


def plot_heatmap(data, destination_folder, filename, colorbar=True):
    fig = plt.figure()
    ax = sns.heatmap(data, yticklabels=False, xticklabels=False)#.set(title='heatmap of '+title)
    ax.tick_params(left=False, bottom=False) 
    if colorbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
    plt.show()
    fig.savefig(os.path.join(destination_folder, filename),bbox_inches='tight',dpi=300)


def plot_singlelink_distance_between_species(tree, folder_name, args, classes):
    destination_folder=os.path.join(args.log_dir,folder_name)
    nodes_count = 2**(args.depth+1)-1

    # maps leaf_id (int) to decision_path that leads to it
    leaf_id_to_decision_vector = collections.defaultdict(lambda: np.zeros((nodes_count, 1)))

    # maps class_id (int) to set of all leaf_nodes
    class_id_to_leaf = collections.defaultdict(set)

    for leaf in tree.leaves:
        class_id_to_leaf[torch.argmax(leaf.distribution()).item()].add(leaf)
        path = tree.path_to(leaf)[:-1] # excluding leaf node
        decision_vector = leaf_id_to_decision_vector[leaf.index]
        for node in path:
            if node.r in path:
                decision_vector[node.index] = 1
            elif node.l in path:
                decision_vector[node.index] = -1
    
    heatmap = np.zeros((len(classes), len(classes)))
    class_ids = list(class_id_to_leaf.keys())
    for i in range(len(class_ids)):
        for j in range(i+1, len(class_ids)):
            class_id_1, class_id_2 = class_ids[i], class_ids[j]
            vector_set1 = [leaf_id_to_decision_vector[leaf.index] for leaf in class_id_to_leaf[class_id_1]]
            vector_set2 = [leaf_id_to_decision_vector[leaf.index] for leaf in class_id_to_leaf[class_id_2]]
            distance = singlelink_distance_between_vectors(vector_set1, vector_set2)
            heatmap[class_id_1][class_id_2] = heatmap[class_id_2][class_id_1] = distance

    # fig = plt.figure()
    # ax = sns.heatmap(heatmap, yticklabels=False, xticklabels=False)#.set(title='heatmap of '+title)
    # ax.tick_params(left=False, bottom=False) 
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=15)
    # plt.show()
    # fig.savefig(os.path.join(destination_folder, 'prototree_species_distances.png'),bbox_inches='tight',dpi=300)

    plot_heatmap(heatmap, destination_folder, filename='prototree_species_distances.png', colorbar=True)

    pd.DataFrame(heatmap).to_csv(os.path.join(destination_folder, 'prototree_species_distances.csv'))


def plot_phylodistance(phylo_filepath, destination_folder):
    tree = Tree(phylo_filepath, format=1)
    leaves = tree.get_leaves()
    leaf_names = sorted([node.name for node in leaves])
    leaves = [leaf for leaf, _ in sorted(zip(leaves, leaf_names), key=lambda x: x[1])] # sort leaves based on leaf_name
    phyl_dist_df = pd.DataFrame(np.zeros((len(leaf_names), len(leaf_names))), index=leaf_names, columns=leaf_names)
    for i in range(len(leaf_names)):
        for j in range(i+1, len(leaf_names)):
            phyl_dist = tree.get_distance(target=leaf_names[i], target2=leaf_names[j], topology_only=False)
            phyl_dist_df.loc[leaf_names[i].name, leaf_names[j].name] = phyl_dist
            
            phyl_dist = tree.get_distance(target=leaf_names[j], target2=leaf_names[i], topology_only=False)
            phyl_dist_df.loc[leaf_names[j].name, leaf_names[i].name] = phyl_dist

    phyl_dist_np = phyl_dist_df.to_numpy()

    plot_heatmap(phyl_dist_np, destination_folder, filename='phylogenetic_species_distances.png', colorbar=True)

    phyl_dist_df.to_csv(os.path.join(destination_folder, 'phylogenetic_species_distances.csv'))

    

# visualize_each_path(tree, 'pruned_and_projected', args, classes)

# distance_between_species(tree, 'pruned_and_projected', args, classes)

# visualize_each_path_with_input(tree, 'pruned_and_projected', args, classes)

plot_singlelink_distance_between_species(tree, 'pruned_and_projected', args, classes)

plot_phylodistance(phylo_filepath="analysis/1_tree-consensus-Hacket-AllSpecies.phy",
                   destination_folder="pruned_and_projected")

# upsample_local(tree: ProtoTree,
#                  sample: torch.Tensor,
#                  sample_dir: str,
#                  decision_path: list,
#                  args: argparse.Namespace)

# upsample_local(tree: ProtoTree,
#                  sample: torch.Tensor,
#                  sample_dir: str,
#                  folder_name: str,
#                  img_name: str,
#                  decision_path: list,
#                  args: argparse.Namespace)
                
      

# total = 0
# for l in class_to_leaves:
#     total += len(class_to_leaves[l])
#     print(len(class_to_leaves[l]))

# breakpoint()

# for leaf in tree.leaves:
#     path_to_leaf = tree.path_to(leaf)
    
#     for node in path_to_leaf:
#         proto_image = Image.open(glob.glob(os.path.join(node_image_path, '*_'+str(node.index)+'_*.jpg'))[0])

