import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import supervision as sv
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image, target_size=(640, 640)):
    """Resize image to the target size."""
    return np.array(Image.fromarray(image).resize(target_size))

def process_images(folder_path, mask_generator):
    """Processes all images in a folder and generates masks."""
    # Get all jpg images from the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, image_file)
        image = np.array(Image.open(image_path).convert("RGB"))

        # Preprocess the image (resize to 640x640)
        resized_image = preprocess_image(image)

        # Generate masks using SAM2
        result = mask_generator.generate(image)

        # Convert to supervision detection format
        detections = sv.Detections.from_sam(sam_result=result)  

        # Annotate masks on the image
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated_image = image.copy()
        annotated_image = mask_annotator.annotate(annotated_image, detections=detections)

        # Display the annotated image
        plt.figure(figsize=(10, 10))
        plt.imshow(annotated_image)
        plt.title(f"Image {idx} - {image_file}")
        plt.axis('off')
        plt.show()
        plt.close()


if __name__ == "__main__":
    # Configuration for SAM2 model
    checkpoint = "checkpoints/floorSeg_checkpoint_epoch500.pt"  # floorSeg_AdaptiveRandomInvertAPI_500
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"  # Update with your config file

    # Initialize SAM2 model
    try:
        print("Entered try block")
        sam2 = build_sam2(model_cfg, checkpoint, device="cuda")
    except Exception as e:
        print(f"Error initializing SAM2: {e}")
        raise

    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # Path to the folder containing images
    folder_path = "propall_testdata/testset2"  # Update with the path to your image folder

    # Run inference
    try:
        process_images(folder_path, mask_generator)
    except Exception as e:
        print(f"Error: {e}")

# TODOs:
# Black and White Inversion Augmentation (for black background)
# Cropping based augmentations to focus on floorplan only
# Wall Detections + Segmentation maps => Try to minimise gaps when both of these inputs overlap (hoping for a contrastive loss or something similar, aim: stretch segments to fitwalls, maybe help in better floorplans )