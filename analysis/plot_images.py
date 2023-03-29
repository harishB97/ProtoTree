import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image, ImageDraw, ImageFont
import random

# Define the path to the CUB dataset
data_dir = "/home/harishbabu/data/CUB_190_split/official/CUB_200_2011/train_segmented_imagenet_background_bb_crop_256"

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image, ImageDraw, ImageFont
import random
import re

# # Define the path to the CUB dataset
# data_dir = "/path/to/cub/dataset"

# Define the transform to resize all images to 224x224 pixels
transform = transforms.Compose([
    transforms.Resize((224, 224))
])

# Create a PyTorch dataset object from the CUB dataset
dataset = ImageFolder(data_dir, transform=transform)

# Get the class names and number of classes in the dataset
class_names = dataset.classes

# Select 5 random classes from the dataset
random_classes = random.sample(class_names, 5)

# Set up the final image by creating a blank canvas
canvas_width = 10 * 224
canvas_height = (5 * 224) + (5 * 10) + 50  # add some extra height for gaps and class labels
canvas = Image.new(mode="RGB", size=(canvas_width, canvas_height), color=(255, 255, 255))

# Set up a font for the class names
font = ImageFont.truetype("/home/harishbabu/projects/ProtoTree/analysis/arial.ttf", 24)

# Iterate over each selected class and create a row of 10 images from that class
for i, class_name in enumerate(random_classes):
    # Get the indices of all images in the current class
    class_indices = [j for j in range(len(dataset)) if dataset.classes[dataset.targets[j]] == class_name]
    
    # Select 10 random indices from the class
    sample_indices = torch.randint(len(class_indices), size=(10,))
    
    # Create a list to hold the images for this row
    row_images = []
    
    # Iterate over the 10 images and add them to the row
    for idx in sample_indices:
        img, _ = dataset[class_indices[idx]]
        row_images.append(img)
    
    # Paste the images onto the canvas with a small gap between them
    x_offset = 20
    y_offset = (i * 224) + (i * 10) + 50  # add some padding for the class labels
    for img in row_images:
        canvas.paste(img, (x_offset + 5, y_offset + 5))
        x_offset += 229  # increase x_offset by 229 (224 + 5) to leave some gap between images
    
    # Get the class name without numbers or dots
    class_name_clean = re.sub("[0-9\.]+", "", class_name)
    print(class_name_clean)
    
    # Add the class name to the left of the first image in the row
    # draw = ImageDraw.Draw(canvas)
    # class_label_width, class_label_height = draw.textsize(class_name_clean, font=font)
    
    # # Rotate the text to make it vertical
    # class_label_img = Image.new(mode="RGB", size=(class_label_height, class_label_width), color=(255, 255, 255))
    # class_label_draw = ImageDraw.Draw(class_label_img)
    # class_label_draw.text((0, 0), class_name_clean, font=font, fill=(0, 0, 0), align="center")
    # class_label_img = class_label_img.rotate(90, expand=True)
    
    # # Paste the class name onto the canvas
    # canvas.paste(class_label_img, (0, y_offset + 112 - (class_label_height // 2)))
    
# # Save the final image
# canvas.save("final_image.png")

# Save the final plot to a file
canvas.save("/home/harishbabu/projects/ProtoTree/analysis/cub_plot.png")