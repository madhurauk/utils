import torch
import pdb
# from kazuto_main_gc import KazutoMain
import json
import os
import glob
import random
from pathlib import Path

import sys
sys.path.append('/srv/share3/mummettuguli3/code/')
from utils.kazuto_main_gc import KazutoMain

with open('imagenet_class_index.json') as f: #is opening file in thesis folder
  data = json.load(f)

class GCUtil:
    def generate_grad_cam(self, model, arch, epoch: str, class_list, layer_name, output_dir, dataset, valdir):
        # eg output_dir : 'GRADCAM_MAPS/resnet18/'
        # with open('imagenet_class_index.json') as f:
        #     data = json.load(f)
        kazuto = KazutoMain()
        classes = kazuto.get_classtable(dataset)
        # pdb.set_trace()
        if dataset == "imagenet":
            for class_name in class_list:
                class_index = classes.index(class_name)
                class_folder_name = data[str(class_index)][0] # eg: n02007558
                folder_files_pattern = os.path.join(valdir, class_folder_name, '*.*')
                image_paths = random.sample(glob.glob(folder_files_pattern),2) # images selected randomly from the folder
                class_folder_path = os.path.join(output_dir, class_folder_name) # 'GRADCAM_MAPS/resnet18/n02007558'
                Path(class_folder_path).mkdir(parents=True, exist_ok=True)
                image_folder_inside_class_folder_list = []
                for j in image_paths:
                  image_name = j.split('/')[-1].split(".")[0] # 'ILSVRC2012_val_00007619'
                  image_folder_inside_class_folder_path = os.path.join(class_folder_path, image_name) # 'GRADCAM_MAPS/resnet18/n02007558/ILSVRC2012_val_00007619'
                  image_predicted_class_folder = os.path.join(image_folder_inside_class_folder_path, 'predicted_class') # 'GRADCAM_MAPS/resnet18/n02007558/ILSVRC2012_val_00007619/predicted_class'
                  image_ground_truth_folder = os.path.join(image_folder_inside_class_folder_path, 'ground_truth') # 'GRADCAM_MAPS/resnet18/n02007558/ILSVRC2012_val_00007619/ground_truth'
                  Path(image_folder_inside_class_folder_path).mkdir(parents=True, exist_ok=True)
                  Path(image_predicted_class_folder).mkdir(parents=True, exist_ok=True)
                  Path(image_ground_truth_folder).mkdir(parents=True, exist_ok=True)
                  image_folder_inside_class_folder_list.append(image_folder_inside_class_folder_path)
                if int(epoch)<10:
                  epoch = '0'+epoch
                # kazuto.demo1(image_paths, layer_name, model, arch+"-{ep:02}".format(ep=epoch), image_folder_inside_class_folder_list, dataset, class_index, class_name)
                kazuto.demo1(image_paths, layer_name, model, arch+"-"+epoch, image_folder_inside_class_folder_list, dataset, class_index, class_name, epoch)