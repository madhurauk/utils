import torch
import pdb
# from kazuto_main_gc import KazutoMain
import json
import os
import glob
import random
from pathlib import Path
import os.path as osp
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.append('/srv/share3/mummettuguli3/code/')
from utils.kazuto_main_gc import KazutoMain
from utils.image_helper import ImageHelper

# with open('imagenet_class_index.json') as f: #this is opening file in thesis folder
#     data = json.load(f)

with open('dataset_subset.json') as f2: #this is opening file in thesis folder
    subset_image_paths = json.load(f2)

font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Medium.ttf", 18)

class GCUtil:
    def __init__(self, dataset):
        self.image_paths_for_classes = {}
        self.output_folder_paths_for_classes = {}
        self.kazuto = KazutoMain()
        self.image_helper = ImageHelper()
        if(dataset == "imagenet"):
            with open('imagenet_class_index.json') as f: #this is opening file in thesis folder
                self.data = json.load(f)
    
    def create_tiny_dataset(self, valdir, sample_count):
        for folder in os.listdir(valdir):
            folder_path = os.path.join(valdir, folder, '*.*')
            image_paths = random.sample(glob.glob(folder_path),sample_count)
            self.image_paths_for_classes[folder] = image_paths #folder is the class_name
        with open('dataset_subset.json', 'w', encoding='utf-8') as f:
            json.dump(self.image_paths_for_classes, f, ensure_ascii=False, indent=4)

    def generate_grad_cam(self, model, arch, epoch: str, class_list, layer_name, dataset):
        # eg output_dir : 'GRADCAM_MAPS/resnet18/'
        classes = self.kazuto.get_classtable(dataset)
        if dataset == "imagenet":
            for class_name in class_list:
                class_index = classes.index(class_name)
                image_paths = self.image_paths_for_classes.get(class_name)
                pdb.set_trace()
                output_paths = self.output_folder_paths_for_classes.get(class_name)

                if int(epoch)<10:
                    epoch = '0'+epoch
                
                # kazuto.demo1(image_paths, layer_name, model, arch+"-{ep:02}".format(ep=epoch), image_folder_inside_class_folder_list, dataset, class_index, class_name)
                self.kazuto.demo1(image_paths, layer_name, model, arch+"-"+epoch, output_paths, dataset, class_index, class_name, epoch)
                # for image_path in image_paths:
                #     self.kazuto.demo1(image_path, layer_name, model, arch+"-"+epoch, output_paths, dataset, class_index, class_name, epoch)
    
    def generate_grad_cam_2(self, model, arch, epoch: str, class_list, layer_name, dataset):
        # eg output_dir : 'GRADCAM_MAPS/resnet18/'
        classes = self.kazuto.get_classtable(dataset)
        # if dataset == "imagenet":
        for class_name in class_list:
            class_index = classes.index(class_name)
            if dataset == "imagenet":
                class_folder_name = self.data[str(class_index)][0]
            else:
                class_folder_name = class_name
            image_paths = self.image_paths_for_classes.get(class_folder_name)
            output_paths = self.output_folder_paths_for_classes.get(class_name)

            if int(epoch)<10:
                epoch_ = '0'+epoch
            else:
                epoch_ = epoch
            
            # kazuto.demo1(image_paths, layer_name, model, arch+"-{ep:02}".format(ep=epoch), image_folder_inside_class_folder_list, dataset, class_index, class_name)
            self.kazuto.demo1(image_paths, layer_name, model, arch+"-"+epoch_, output_paths, dataset, class_index, class_name, epoch_)

    def create_output_folder_2(self, output_dir, dataset, class_list):
        classes = self.kazuto.get_classtable(dataset)
        self.image_paths_for_classes = subset_image_paths
        # if dataset == "imagenet":
        for class_name in class_list:
            if dataset == "imagenet":
                class_index = classes.index(class_name)
                class_folder_name = self.data[str(class_index)][0] # eg: n02007558
            else:
                class_folder_name = class_name
            image_paths = subset_image_paths[str(class_folder_name)]

            #create output folder for the class
            class_folder_path = os.path.join(output_dir, class_folder_name) # 'GRADCAM_MAPS/resnet18/folder1/n02007558'
            Path(class_folder_path).mkdir(parents=True, exist_ok=True)

            #create folder for each image inside the class folder, then create 'predicted' and 'ground_truth' folder inside each folder
            self.output_folder_paths_for_classes[class_name] = []
            for j in image_paths:
                image_name = j.split('/')[-1].split(".")[0] # 'ILSVRC2012_val_00007619'
                image_folder_inside_class_folder_path = os.path.join(class_folder_path, image_name) # 'GRADCAM_MAPS/resnet18/n02007558/ILSVRC2012_val_00007619'
                image_predicted_class_folder = os.path.join(image_folder_inside_class_folder_path, 'predicted_class') # 'GRADCAM_MAPS/resnet18/n02007558/ILSVRC2012_val_00007619/predicted_class'
                image_ground_truth_folder = os.path.join(image_folder_inside_class_folder_path, 'ground_truth') # 'GRADCAM_MAPS/resnet18/n02007558/ILSVRC2012_val_00007619/ground_truth'
                Path(image_folder_inside_class_folder_path).mkdir(parents=True, exist_ok=True)
                Path(image_predicted_class_folder).mkdir(parents=True, exist_ok=True)
                Path(image_ground_truth_folder).mkdir(parents=True, exist_ok=True)
                self.output_folder_paths_for_classes[class_name].append(image_folder_inside_class_folder_path) 

                image = self.image_helper.open_image(j)
                image_copy = image.copy()
                self.image_helper.add_text_save_file(image_copy, class_name, osp.join(
                    image_folder_inside_class_folder_path,
                    "{}.png".format(
                        class_name
                    ),
                ))

    def create_output_folder(self, output_dir, dataset, class_list, valdir, sample_count):
        classes = self.kazuto.get_classtable(dataset)
        if dataset == "imagenet":
            for class_name in class_list:
                class_index = classes.index(class_name)
                image_paths = self.image_paths_for_classes.get(class_name)
                if image_paths is None:
                  #select images for this class randomly  
                  class_folder_name = self.data[str(class_index)][0] # eg: n02007558
                  folder_files_pattern = os.path.join(valdir, class_folder_name, '*.*')
                  random.seed(10)
                  image_paths = random.sample(glob.glob(folder_files_pattern),sample_count) # images selected randomly from the folder
                  self.image_paths_for_classes[class_name] = image_paths
                  
                  #calculate timestamp
                  import calendar
                  import time
                  
                  # gmt stores current gmtime
                  gmt = time.gmtime()
                  print("gmt:-", gmt)
                  
                  # ts stores timestamp
                  ts = calendar.timegm(gmt)
                  print("timestamp:-", ts)

                  #create output folder for the class
                  class_folder_path = os.path.join(output_dir, str(ts), class_folder_name) # 'GRADCAM_MAPS/resnet18/timestamp/n02007558'
                  Path(class_folder_path).mkdir(parents=True, exist_ok=True)

                  #create folder for each image inside the class folder, then create 'predicted' and 'ground_truth' folder inside each folder
                  self.output_folder_paths_for_classes[class_name] = []
                  for j in image_paths:
                    image_name = j.split('/')[-1].split(".")[0] # 'ILSVRC2012_val_00007619'
                    image_folder_inside_class_folder_path = os.path.join(class_folder_path, image_name) # 'GRADCAM_MAPS/resnet18/n02007558/ILSVRC2012_val_00007619'
                    image_predicted_class_folder = os.path.join(image_folder_inside_class_folder_path, 'predicted_class') # 'GRADCAM_MAPS/resnet18/n02007558/ILSVRC2012_val_00007619/predicted_class'
                    image_ground_truth_folder = os.path.join(image_folder_inside_class_folder_path, 'ground_truth') # 'GRADCAM_MAPS/resnet18/n02007558/ILSVRC2012_val_00007619/ground_truth'
                    Path(image_folder_inside_class_folder_path).mkdir(parents=True, exist_ok=True)
                    Path(image_predicted_class_folder).mkdir(parents=True, exist_ok=True)
                    Path(image_ground_truth_folder).mkdir(parents=True, exist_ok=True)
                    self.output_folder_paths_for_classes[class_name].append(image_folder_inside_class_folder_path) 

                    image = self.image_helper.open_image(j)
                    image_copy = image.copy()
                    self.image_helper.add_text_save_file(image_copy, class_name, osp.join(
                        image_folder_inside_class_folder_path,
                        "{}.png".format(
                            class_name
                        ),
                    ))
        else:
            for class_folder_name in class_list:      
                folder_files_pattern = os.path.join(valdir, class_folder_name, '*.*')
                image_paths = random.sample(glob.glob(folder_files_pattern),sample_count) # images selected randomly from the folder
                self.image_paths_for_classes[class_folder_name] = image_paths

                class_folder_path = os.path.join(output_dir, class_folder_name)
    
    def initiate_create_gif_2(self, dataset, output_dir, class_list):
        classes = self.kazuto.get_classtable(dataset)
        # if dataset == "imagenet":
        for class_name in class_list:
            # pdb.set_trace()
            if dataset == "imagenet":
                class_index = classes.index(class_name)
                class_folder_name = self.data[str(class_index)][0] # eg: n02007558
            else:
                class_folder_name = class_name
            class_folder_path = osp.join(output_dir, class_folder_name)
            for j in os.listdir(class_folder_path):
                # pdb.set_trace()
                predicted_gcam_path = osp.join(class_folder_path,
                        j, # eg: ILSVRC2012_val_00024268
                        'predicted_class'
                        )
                
                filenames = []
                files = []

                for i in os.listdir(predicted_gcam_path):
                    filepath = os.path.join(predicted_gcam_path,i)
                    if os.path.isfile(filepath) and '-gradcam-'in i:                                                                           
                        files.append(filepath)
                        filenames.append(i)

                files.sort()
                filenames.sort()

                frames = []

                for i,filename in enumerate(files):
                    im = Image.open(filename)
                    frames.append(im)

                frames[0].save(osp.join(class_folder_path,
                        j,
                        'predicted.gif'
                        ), format='GIF',
                            append_images=frames[1:], save_all=True, duration=240, loop=0)

                ground_truth_gcam_path = osp.join(class_folder_path,
                        j,
                        'ground_truth'
                        )
                
                filenames = []
                files = []

                for i in os.listdir(ground_truth_gcam_path):
                    filepath = os.path.join(ground_truth_gcam_path,i)
                    if os.path.isfile(filepath) and '-gradcam-'in i:                                                                           
                        files.append(filepath)
                        filenames.append(i)

                files.sort()
                filenames.sort()

                frames = []

                for i,filename in enumerate(files):
                    im = Image.open(filename)
                    frames.append(im)

                frames[0].save(osp.join(class_folder_path,
                        j,
                        'ground_truth.gif'
                        ), format='GIF',
                            append_images=frames[1:], save_all=True, duration=240, loop=0)
    
    def initiate_create_gif(self, dataset, class_list):
        if dataset == "imagenet":
            for class_name in class_list:
                output_paths = self.output_folder_paths_for_classes.get(class_name)
                for j in output_paths:
                    predicted_gcam_path = osp.join(
                            j,
                            'predicted_class'
                            )
                    # self.create_gif(predicted_gcam_path, osp.join(
                    #         j,
                    #         'predicted.gif'
                    #         ))
                    filenames = []
                    files = []

                    for i in os.listdir(predicted_gcam_path):
                        filepath = os.path.join(predicted_gcam_path,i)
                        if os.path.isfile(filepath) and '-gradcam-'in i:                                                                           
                            files.append(filepath)
                            filenames.append(i)

                    files.sort()
                    filenames.sort()

                    frames = []

                    for i,filename in enumerate(files):
                        im = Image.open(filename)
                        frames.append(im)

                    frames[0].save(osp.join(
                            j,
                            'predicted.gif'
                            ), format='GIF',
                                append_images=frames[1:], save_all=True, duration=240, loop=0)

                    ground_truth_gcam_path = osp.join(
                            j,
                            'ground_truth'
                            )
                    # self.create_gif(ground_truth_gcam_path, osp.join(
                    #         j,
                    #         'ground_truth.gif'
                    #         ))
                    filenames = []
                    files = []

                    for i in os.listdir(ground_truth_gcam_path):
                        filepath = os.path.join(ground_truth_gcam_path,i)
                        if os.path.isfile(filepath) and '-gradcam-'in i:                                                                           
                            files.append(filepath)
                            filenames.append(i)

                    files.sort()
                    filenames.sort()

                    frames = []

                    for i,filename in enumerate(files):
                        im = Image.open(filename)
                        frames.append(im)

                    frames[0].save(osp.join(
                            j,
                            'ground_truth.gif'
                            ), format='GIF',
                                append_images=frames[1:], save_all=True, duration=240, loop=0)
    
    def create_gif(self, path, gif_filename):
        '''
        not using this at the moment '''
        filenames = []
        files = []

        for i in os.listdir(path):
            filepath = os.path.join(path,i)
            if os.path.isfile(filepath) and '-gradcam-'in i:                                                                           
                files.append(filepath)
                filenames.append(i)

        files.sort()
        filenames.sort()

        frames = []

        for i,filename in enumerate(files):
            im = Image.open(filename)
            draw = ImageDraw.Draw(im)
            draw.text((0, 0),str(i+1)+":"+filename.split("-")[-1],(255,255,255),font=font)
            frames.append(im)

        frames[0].save(gif_filename, format='GIF',
                    append_images=frames[1:], save_all=True, duration=240, loop=0)
                    
