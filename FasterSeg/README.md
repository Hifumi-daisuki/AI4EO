# OrbitalAI_TelePIX+KIOST

## Requirements
- Ubuntu 18.04
- CUDA 10.0
- Cudnn 7.6.5

Install dependencies  
```
pip3 install -r requirements.txt
```

## Dataset preparation
Our dataset is available [here](https://drive.google.com/drive/folders/1Xyh_aJZCDGTDpfz0nXOO0BLJGtuwUenX?usp=drive_link)  
Put image files in ./data folder.  
Revise train, val, train_val, test.txt of ./data.  

## 0. Test model
* Our pretrained weight is already uploaded in ./train/weight0.pt. 
* Set `C.is_eval = True` in `config_train.py`.
* Set the name of the searched folder as `C.load_path = "fasterseg"` in `config_train.py`.
* Download the pretrained weights of the [teacher](https://drive.google.com/file/d/168HtgNnY9OdCz5Z6FWxoJr-gd5EtS5Sp/view?usp=sharing) and [student](https://drive.google.com/file/d/1O56HnA0ug2M3K4SR3_AUzIs0wegy9BX6/view?usp=sharing) and put them into folder `train/fasterseg`.
<!-- * set the name of pretrained directory as `C.eval_path = "/path/to/pretrained/models/"` in `config_train.py`. -->
* Start the evaluation process:
```bash
python train.py
```

### 1. Search
```bash
cd search
```
#### 1.1 Pretrain the supernet
We first pretrain the supernet without updating the architecture parameter for 20 epochs.
* Set `C.pretrain = True` in `config_search.py`.
* Start the pretrain process:
```bash
CUDA_VISIBLE_DEVICES=0 python train_search.py
```
* The pretrained weight will be saved in a folder like ```FasterSeg/search/search-pretrain-256x512_F12.L16_batch3-20200101-012345```.

#### 1.2 Search the architecture
We start the architecture searching for 30 epochs.
* Set the name of your pretrained folder (see above) `C.pretrain = "search-pretrain-256x512_F12.L16_batch3-20200101-012345"` in `config_search.py`.
* Start the search process:
```bash
CUDA_VISIBLE_DEVICES=0 python train_search.py
```
* The searched architecture will be saved in a folder like ```FasterSeg/search/search-224x448_F12.L16_batch2-20200102-123456```.
* `arch_0` and `arch_1` contains architectures for teacher and student networks, respectively.

### 2. Train from scratch
* `cd FasterSeg/train`
* Copy the folder which contains the searched architecture into `FasterSeg/train/` or create a symlink via `ln -s ../search/search-224x448_F12.L16_batch2-20200102-123456 ./`
#### 2.1 Train the teacher network
* Set `C.mode = "teacher"` in `config_train.py`.
<!-- * uncomment the `## train teacher model only ##` section in `config_train.py` and comment the `## train student with KL distillation from teacher ##` section. -->
* Set the name of your searched folder (see above) `C.load_path = "search-224x448_F12.L16_batch2-20200102-123456"` in `config_train.py`. This folder contains `arch_0.pt` and `arch_1.pth` for teacher and student's architectures.
* Start the teacher's training process:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```
* The trained teacher will be saved in a folder like `train-512x1024_teacher_batch12-20200103-234501`
#### 2.2 Train the student network (FasterSeg)
* Set `C.mode = "student"` in `config_train.py`.
<!-- * uncomment the `## train student with KL distillation from teacher ##` section in `config_train.py` and comment the `## train teacher model only ##` section. -->
* Set the name of your searched folder (see above) `C.load_path = "search-224x448_F12.L16_batch2-20200102-123456"` in `config_train.py`. This folder contains `arch_0.pt` and `arch_1.pth` for teacher and student's architectures.
* Set the name of your teacher's folder (see above) `C.teacher_path = "train-512x1024_teacher_batch12-20200103-234501"` in `config_train.py`. This folder contains the `weights0.pt` which is teacher's pretrained weights.
* Start the student's training process:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```


## Acknowledgement
Mainly from [FasterSeg Offical](https://github.com/VITA-Group/FasterSeg) 

