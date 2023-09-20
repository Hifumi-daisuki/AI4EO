# OrbitalAI_TelePIX+KIOST

## Requirements
```
pip3 install -r requirements.txt
```

## Dataset preparation
Revise train, val, train_val, test.txt of data folder.

## Test model
Test a image from specific directory on the trained model as follows
```
python tools/cityscapes/test_bisenetv2_cityscapes.py --weights_path ./weights/cityscapes/bisenetv2/cityscapes.ckpt  --src_image_path ./data/val/
```

## Train model
Start your training procedure.
```
CUDA_VISIBLE_DEVICES="0" python tools/cityscapes/train_bisenetv2_cityscapes.py
```

## Acknowledgement
Mainly from [bisenetv2-tensorflow](https://github.com/MaybeShewill-CV/bisenetv2-tensorflow) 
