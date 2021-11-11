#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os.path as osp
from utils.image_helper import ImageHelper

# import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)
import pdb
from PIL import Image, ImageDraw, ImageFont
from GPUtil import showUtilization as gpu_usage

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False

class KazutoMain:
    def get_device(self, cuda):
        cuda = cuda and torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        if cuda:
            current_device = torch.cuda.current_device()
            print("Device:", torch.cuda.get_device_name(current_device))
        else:
            print("Device: CPU")
        return device


    def load_images(self, image_paths):
        images = []
        raw_images = []
        print("Images:")
        for i, image_path in enumerate(image_paths):
            print("\t#{}: {}".format(i, image_path))
            image, raw_image = self.preprocess(image_path)
            images.append(image)
            raw_images.append(raw_image)
        return images, raw_images


    def get_classtable(self, dataset):
        classes = []
        if dataset == "imagenet":
            with open("synset_words.txt") as lines:
                for line in lines:
                    line = line.strip().split(" ", 1)[1]
                    line = line.split(", ", 1)[0].replace(" ", "_")
                    classes.append(line)
        else:
            with open("categories_places365.txt") as lines:
                for line in lines:
                    line = line.strip().split(" ", 1)[0]
                    line = line.split("/")[2]
                    classes.append(line)
        return classes


    def preprocess(self, image_path):
        raw_image = cv2.imread(image_path)
        raw_image = cv2.resize(raw_image, (224,) * 2)
        image = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(raw_image[..., ::-1].copy())
        return image, raw_image


    def save_gradient(self, filename, gradient):
        gradient = gradient.cpu().numpy().transpose(1, 2, 0)
        gradient -= gradient.min()
        gradient /= gradient.max()
        gradient *= 255.0
        cv2.imwrite(filename, np.uint8(gradient))


    def save_gradcam(self, filename, gcam, raw_image, text, paper_cmap=False):
        # pdb.set_trace()
        gcam = gcam.cpu().numpy()
        cmap = cm.jet_r(gcam)[..., :3] * 255.0
        if paper_cmap:
            alpha = gcam[..., None]
            gcam = alpha * cmap + (1 - alpha) * raw_image
        else:
            gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        image = np.uint8(gcam)
        # ImageHelper().add_text_save_file(image, text, filename)
        org = (5, 15)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0, 0, 0)                                                                                                                                              
        thickness = 1
        image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imwrite(filename, image)
        # pdb.set_trace()


    def save_sensitivity(self, filename, maps):
        maps = maps.cpu().numpy()
        scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
        maps = maps / scale * 0.5
        maps += 0.5
        maps = cm.bwr_r(maps)[..., :3]
        maps = np.uint8(maps * 255.0)
        maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(filename, maps)


    # @main.command()
    # @click.option("-i", "--image-paths", type=str, multiple=True, required=True)
    # @click.option("-a", "--arch", type=click.Choice(model_names), required=True)
    # @click.option("-t", "--target-layer", type=str, required=True)
    # @click.option("-k", "--topk", type=int, default=3)
    # @click.option("-o", "--output-dir", type=str, default="./results")
    # @click.option("--cuda/--cpu", default=True)
    # def demo1(image_paths, target_layer, arch, topk, output_dir, cuda):
    def demo1(self, image_paths, target_layer, model, arch, output_dir_list, dataset, class_index, class_name, epoch, cuda=True, topk=1):
        """
        Visualize model responses given multiple images
        """

        device = self.get_device(cuda)

        # Synset words
        classes = self.get_classtable(dataset)

        model.eval()

        # print("Memory usage before load_images",)
        # gpu_usage() 
        # Images
        images, raw_images = self.load_images(image_paths)
        images = torch.stack(images).to(device)
        # print("Memory usage after load_images",)
        # gpu_usage() 

        """
        Common usage:
        1. Wrap your model with visualization classes defined in grad_cam.py
        2. Run forward() with images
        3. Run backward() with a list of specific classes
        4. Run generate() to export results
        """

        # =========================================================================
        # print("Vanilla Backpropagation:")

        bp = BackPropagation(model=model)
        probs, ids = bp.forward(images)  # sorted

        # =========================================================================
        print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")
        ### For predicted class
        gcam = GradCAM(model=model)
        _ = gcam.forward(images)
        
        for i in range(topk):

            # Grad-CAM
            # pdb.set_trace()
            gcam.backward(ids=ids[:, [i]])
            regions = gcam.generate(target_layer=target_layer) 

            for j in range(len(images)):
                # pdb.set_trace()
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                # Grad-CAM
                filename = osp.join(
                        output_dir_list[j],
                        'predicted_class',
                        "{}-{}-gradcam-{}-{}.png".format(
                            j, arch, target_layer, classes[ids[j, i]]
                        ),
                    )
                self.save_gradcam(
                    filename=filename,
                    gcam=regions[j, 0],
                    raw_image=raw_images[j],
                    text=epoch+":"+classes[ids[j, i]]
                )
            del regions
        ### For ground truth class
        gcam2 = GradCAM(model=model)
        _2 = gcam2.forward(images) 
        ground_truth_id = torch.full((images.shape[0],1),class_index).to(device)
        # ground_truth_id = torch.tensor(class_index).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
        gcam2.backward(ids=ground_truth_id) 
        regions = gcam2.generate(target_layer=target_layer) 

        for j in range(len(images)):
            # print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Grad-CAM
            self.save_gradcam(
                filename=osp.join(
                    output_dir_list[j],
                    'ground_truth',
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, arch, target_layer, class_name
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
                text=epoch+":"+class_name
            )
        del images
        del raw_images
        del bp
        del probs
        del ids
        del regions
        del gcam
        del gcam2
        del _2
        del _
        torch.cuda.empty_cache()
        # print("memory summary:",torch.cuda.memory_summary(device=None, abbreviated=False))