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
Revise train, val, train_val, test.txt of data folder.  

## Test model
Our pretrained weight is available [here](https://drive.google.com/drive/folders/1rkc5AkHkIUofTwEC9-mi9x2_UIl3C1nd?usp=drive_link)  
Put weight files in ./weights folder.  
Test a image from specific directory on the trained model as follows
```
python tools/cityscapes/test_bisenetv2_cityscapes.py --weights_path ./weights/cityscapes.ckpt  --src_image_path ./data/image/test/
```
Output images will be saved in ./results folder.

## Train model
Start your training procedure.
```
python tools/cityscapes/train_bisenetv2_cityscapes.py
```

## Acknowledgement
Mainly from [bisenetv2-tensorflow](https://github.com/MaybeShewill-CV/bisenetv2-tensorflow) 
