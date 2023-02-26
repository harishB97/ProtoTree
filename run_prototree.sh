#!/bin/bash

#SBATCH --account=mabrownlab
#SBATCH --partition=dgx_normal_q
#SBATCH --time=1-00:00:00 
#SBATCH --gres=gpu:3
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ./SLURM/slurm-%j.out


echo start load env and run python

module reset
module load Anaconda3/2020.11
source activate taming3
module reset
source activate taming3
which python


python main_tree.py --epochs 100 \
                    --log_dir ./runs/010-cub_190_imgnet_224-dth=9-ep=100 \
                    --dataset CUB-224-imgnetmean \
                    --lr 0.001 \
                    --lr_block 0.001 \
                    --lr_net 1e-5 \
                    --num_features 256 \
                    --depth 9 \
                    --net resnet50_inat \
                    --freeze_epochs 30 \
                    --milestones 60,70,80,90,100 \
                    --gpus 0,1,2 #\
                    # --state_dict_dir_net '/home/harishbabu/projects/ProtoTree/runs/005-cub_190_imgnet_224-dth=9-ep=100/checkpoints/latest/'

exit;




# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt