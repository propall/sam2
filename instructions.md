### 1. Installation

```bash
conda create -n floorSeg python=3.10
conda activate floorSeg

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .

pip install -e ".[notebooks]"

pip install submitit tensordict tensorboard fvcore pandas pycocotools

pip install nvitop
```

### 2. Datasets used

| Name     | Origin URL     | Total | Resize |
| :------: | :-----: | :---: | :---: |
| [Set1](https://drive.google.com/file/d/16yvERPJfAUUXhhKoyRwJh_2yT0Ry1dm2/view?usp=sharing)     | https://universe.roboflow.com/usama-mahtab-lj8dy/floored-detection-a5ehw | 2583 | different sizes |
| [Set2](https://drive.google.com/file/d/1gPLlSQ129Ol6zfYpGO5Xqa4XWK7f2PjW/view?usp=sharing)     | https://universe.roboflow.com/rohini/segv2/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true | 438 | 640 |

Create a folder ```floorplan_datasets``` inside sam2 directory and move all the unzipped dataset folders to it.

### 3. Download Checkpoints
```bash
./checkpoints/download_ckpts.sh
```
Move the model pt files into "checkpoints" directory.

### 4. Dataset Formats
4.1 [Segment Anything 1 Billion (SA-1B)](https://ai.meta.com/datasets/segment-anything/)
    This dataset is the dataset for original SAM model. It is the format for any image based downstream tasks.

```
    {
    "image"                 : image_info,
    "annotations"           : [annotation],
    }

    image_info {
        "image_id"              : int,              # Image id
        "width"                 : int,              # Image width
        "height"                : int,              # Image height
        "file_name"             : str,              # Image filename
    }

    annotation {
        "id"                    : int,              # Annotation id
        "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
        "area"                  : int,              # The area in pixels of the mask
        "segmentation"          : seg_dict,         # Mask saved in COCO RLE format.
        
        
        "predicted_iou"         : float,            # The model's own prediction of the mask's quality
        "stability_score"       : float,            # A measure of the mask's quality
        "crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
        "point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
    }

    seg_dict {
        "counts"                : str,              # Mask encoded in Run-Length Encoding (RLE) format
        "size"                  : [w, h]
    }
```

The format expected for SAM(or SAM2) training:
```
    SAM_dataset
    |______train
            |______img1.jpg
            |______img1.json
            |______img2.jpg
            |______img2.json
    |______test
            |______img1.jpg
            |______img1.json
            |______img2.jpg
            |______img2.json
    |______valid
            |______img1.jpg
            |______img1.json
            |______img2.jpg
            |______img2.json
```

Note: This format is different from COCO Segmentation format which is of the below format:

```
    COCO_dataset
    |______train
            |______img1.jpg
            |______img2.jpg
            |______annotations.json
    |______test
            |______img1.jpg
            |______img2.jpg
            |______annotations.json
    |______valid
            |______img1.jpg
            |______img2.jpg
            |______annotations.json
```

How RLE encoding works in Pycocotools?
- RLE is used to encode binary masks for image segmentation tasks by default
- Pixels belonging to foreground are 1 and background are 0.
- 2D binary mask is flattened into a 1D array (vector) in **column-major order** (also known as Fortran order).
- Check if the order is column major or row major clearly before RLE encoding.

Recording RLE:
* The counts start with the number of zeros (background pixels) at the beginning of the array.
* The counts alternate between the number of zeros and the number of ones.
* If the mask starts with a one, the first count is zero to indicate there are no zeros before the first run of ones.
* Eg: [0 0 1 1 1 0 1] => [2311] (starts with 0, so 2 zeroes, 3 ones, 1 zero and 1 one).
* Eg: [1 1 1 1 1 1 0] => [061] (starts with 1, so 0 zeroes, 6 ones and 1 zero)


More explanation regarding RLE is given in <https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py>

RLE for multiclass Segmentation:
- RLE can be adapted to deal with multiclass segmentations by simply adding a 'values' key to the dictionary used. This values tells what is the class and counts tells how many times that class is repeated.
- Eg: The below multiclass 2D array is encoded as follows:

```
array = np.array([
    [0, 0, 1, 1, 0],
    [3, 0, 1, 1, 0],
    [3, 3, 1, 1, 0],
    [3, 3, 0, 0, 0]
])

rle = {
    'size': (4, 5),
    'values': (0, 3, 0, 3, 1, 0, 1, 0)
    'counts': (1, 3, 2, 2, 3, 1, 3, 5)
}

```



4.2 [Segment Anything Video (SA-V)](https://ai.meta.com/datasets/segment-anything-video/)


### 5. Training SAM2

The original source file for training is <github.com/facebookresearch/sam2/blob/main/training/README.md> 

How to modify training configuration file for image only dataset: <github.com/facebookresearch/sam2/issues/347>

```TorchTrainMixedDataset````(training/datasets/sam2_datasets.py)is a custom dataloader that handles multiple datasets in a single training pipeline, allowing for different batch sizes and probabilities of sampling from each dataset.

Add configs folder and add the training config file.

To train the model, use

```bash
python training/train.py -c ../configs/Set1_finetune.yaml
python training/train.py -c ../configs/Set2_finetune.yaml --use-cluster 0
```
".." is necessary in config path and if not gives the error in <github.com/facebookresearch/sam2/issues/177>. Couldn't figure out why though.

<github.com/facebookresearch/sam2/pull/207> has a pull request with comments for yaml file.

### 6. Visualise with Tensorboard

```bash
tensorboard --bind_all --logdir ./sam2_logs/tensorboard/
```

Notes:
Segmentation Format:

The "counts" field represents the mask in Run-Length Encoding (RLE) format.
The "size" field specifies the dimensions of the mask (height and width).

pin_memory:
Pinned memory, also known as page-locked memory, is a type of memory allocation in a system’s RAM that prevents it from being swapped out to disk. pin_memory=True in DataLoader is used to load data quickly to GPU. This is beneficial if the system has ample RAM.



conda install -c anaconda ipykernel
python -m ipykernel install --user --name=<your_conda_env>


To delete env or uninstall pytorch
```bash
conda remove --name ENV_NAME --all
conda remove pytorch torchvision torchaudio
```

To check if GPU is working
```bash
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
print(torch.__version__)
```