{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shutil\n",
    "import os\n",
    "\n",
    "src_folders = ['001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross', '045.Northern_Fulmar']\n",
    "src_dirs = [os.path.join('/fastscratch/harishbabu/data/CUB_200_2011/dataset/train_corners', folder_name) for folder_name in src_folders]\n",
    "tgt_dir = '/fastscratch/haris/hbabu/data/CUB_subset_1/train_corners/1.node1'\n",
    "\n",
    "for src_dir in src_dirs:\n",
    "    shutil.copy(src_dir, tgt_dir)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
