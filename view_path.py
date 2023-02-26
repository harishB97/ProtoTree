from prototree.prototree import ProtoTree
from PIL import Image
import glob
import os
import torch
from util.data import get_dataloaders
# from util.phylogeny import PhylogenyCUB

from util.args import load_args

# from prototree import ProtoTree
args = load_args('/home/harishbabu/projects/ProtoTree/runs/010-cub_190_imgnet_224-dth=9-ep=100/metadata')

log_dir = args.log_dir
trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args)

tree = ProtoTree.load(os.path.join(log_dir, 'checkpoints/latest'))

node_image_path = os.path.join(log_dir, 'pruned_and_projected/node_vis')

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

        
                


        

# total = 0
# for l in class_to_leaves:
#     total += len(class_to_leaves[l])
#     print(len(class_to_leaves[l]))

# breakpoint()

# for leaf in tree.leaves:
#     path_to_leaf = tree.path_to(leaf)
    
#     for node in path_to_leaf:
#         proto_image = Image.open(glob.glob(os.path.join(node_image_path, '*_'+str(node.index)+'_*.jpg'))[0])

