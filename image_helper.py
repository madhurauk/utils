import cv2
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