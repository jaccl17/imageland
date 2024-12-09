from PIL import Image
import os

def trim_and_resize_images(input_folder, output_folder, target_width, target_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    
    for filename in os.listdir(input_folder): # loop through all files in the input folder

        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".bmp"):
            with Image.open(os.path.join(input_folder, filename)) as img:
		# crops from center evenly on all sides
                left = (img.width - target_width) // 2
                top = (img.height - target_height) // 2
                right = left + target_width
                bottom = top + target_height

                img = img.crop((left, top, right, bottom))
 
                img = img.resize((target_width, target_height), Image.ANTIALIAS)
 
                img.save(os.path.join(output_folder, filename))

input_folder = "/home/unitx/wabbit_playground/estee_lauder/raw_dropper"
output_folder = "/home/unitx/wabbit_playground/estee_lauder/dropper_cropped"
target_width = 95
target_height = 1280

trim_and_resize_images(input_folder, output_folder, target_width, target_height)
