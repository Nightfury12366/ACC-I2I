# ACC-I2I
The pytorch implement of "An Attribute Consistency Constraint Model for End-to-end Image Translation"
<p align="center">
<img src="./doc/s_model.pdf" width="800px"/>
<br></p>

## Dependency
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install tqdm gdown kornia scipy opencv-python dlib moviepy lpips pillow visdom 
```

## Pretrained Models
The pre-trained model can be found in the /checkpoints folder:

| Sefile2Ainme | Face2Ainme | Dog2Cat | Cat2Dog
| :----------- | :--------- | :------ | :------
| /s2a_06	   | /f2a_06    | /d2c_06 | /c2d_06


## Dataset
The dataset we use for training is the [face2anime](https://drive.google.com/file/d/1mYPo5JKZKypfr-lmURt_HGukb77TJ0hC/view?usp=sharing) dataset builded from FFHQ and Danbooru2020. You can also use your own dataset in the following format.
```
└── YOUR_DATASET_NAME
   ├── trainA
       ├── xxx.jpg (name, format doesn't matter)
       ├── yyy.png
       └── ...
   ├── trainB
       ├── zzz.jpg
       ├── www.png
       └── ...
   ├── testA
       ├── aaa.jpg 
       ├── bbb.png
       └── ...
   └── testB
       ├── ccc.jpg 
       ├── ddd.png
       └── ...
```

Sefile2Anime: [selfie2anime](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view?usp=sharing) dataset from UGATIT.
Cat2Dog and Dog2Cat: AFHQ dataset from stargan-v2.

## Test

### Cat-to-Dog
```bash
python test.py --name c2d_06 --direction BtoA --dataroot ./datasets/afhq
```

### Dog-to-Cat
```bash
python test.py --name d2c_06 --dataroot ./datasets/afhq
```

### Selfie2anime
```bash
python test.py --name s2a_06 --dataroot ./datasets/selfie2anime 

```
### Selfie2anime
```bash
python test.py --name f2a_06 --dataroot ./datasets/face2anime 
```

## Some Results

### Compare with baseline
<p align="center">
<img src="./doc/compare_w.pdf" width="800px"/>
<br></p>

### Realtime Video I2I Performance
<p align="center">
<img src="./doc/realtime_video_i2i.pdf" width="800px"/>
<br></p>
