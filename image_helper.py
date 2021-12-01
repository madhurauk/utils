import cv2
import os
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_pdf import PdfPages

class ImageHelper:
    def open_image(self, image_path):
        raw_image = cv2.imread(image_path)
        return raw_image
    
    def add_text_save_file(self, image, text, filename):
        org = (5, 15)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0, 0, 0)                                                                                                                                              
        thickness = 1
        resized_image = cv2.resize(image, (224,224))
        resized_image_with_text = cv2.putText(resized_image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imwrite(filename, resized_image_with_text)
    
    def sort_gcam_images_in_folder(self, folderpath):
        files = []
        for i in os.listdir(folderpath):
            filepath = os.path.join(folderpath,i)
            if os.path.isfile(filepath) and '-gradcam-'in i:                                                                           
                files.append(filepath)
        files.sort()
        return files
    
    def parse_gcam_files_in_folder(self, path, class_type):
#     path = "GRADCAM_MAPS/resnet18/154350/n01614925/"
        folders = next(os.walk(path))[1]
        dict_1 = {}
        dict_2 = {}

        for index, j in enumerate(folders):
            sub_path = os.path.join(path,j, class_type)
            files = self.sort_gcam_images_in_folder(sub_path)
            if index < 5:
                dict_1[j] = files
            else:
                dict_2[j] = files
        return [dict_1,dict_2]
    
    def show_img_3(dict_1):
        rows = 90
        cols = 5
        img_count = 0
        num_images = (rows*cols)

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,300))

        for i,key in enumerate(dict_1):
            for j,file in enumerate(dict_1[key]): 
    #             pdb.set_trace()
                if img_count < num_images:
                    axes[j, i].imshow(imageio.imread(file))
                    img_count+=1
        return fig
    
    def show_img_save_pdf(self, dict_1, pdf_filename, class_type):
        rows = 4
        cols = 5
        img_count = 0
        num_images = (90*5)
        z = 0
        figs = []
        used_axes = []
        
        with PdfPages(pdf_filename) as pdf:
            for z in range(0,90,4):
                fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))

                for i,key in enumerate(dict_1):
                    for j,file in enumerate(dict_1[key][z:z+4]): 
            #             pdb.set_trace()
                        if img_count < num_images:
                            axes[j, i].imshow(imageio.imread(file))
                            axes[j, i].title.set_text("epoch "+str(z+j+1)+":"+file.split("-")[-1].split(".")[0])
                            img_count+=1
                            used_axes.append(axes[j, i])

                ##removing empty axes:
                for ax in axes.reshape(-1):
                    if ax not in used_axes:
                        ax.remove()

                fig.suptitle(class_type+" gradcam maps")
                fig.tight_layout()
                figs.append(fig)
                pdf.savefig()
        return figs

    def view_gcam_generate_pdf(self, path, class_type):
        files = self.parse_gcam_files_in_folder(path, class_type)
        self.show_img_save_pdf(files[0],path+class_type+'_1.pdf',class_type)
        self.show_img_save_pdf(files[1],path+class_type+'_2.pdf',class_type)

