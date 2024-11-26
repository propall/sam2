# SAM 2: Segment Anything in Images

Create Conda Env:
```bash
conda create -n floorSeg python=3.10
conda activate floorSeg
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
Setup SAM2:
<!-- git clone https://github.com/propall/sam2.git -->
```bash
cd sam2
pip install -e .
pip install -e ".[notebooks]"

pip install fvcore submitit tensordict tensorboard pandas pycocotools nvitop supervision
```

To train:
```bash
python training/train.py -c ../configs/Set2_finetune.yaml
```

For inference:
```bash
python test_inference.py
```