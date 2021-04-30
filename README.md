# 训练模型：
先配置好conda环境后，
安装
```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```
以及pytorch
然后拉去代码库中的yolof
Install YOLOF by:
```
python setup.py develop
```
Then link your dataset path to datasets
```
cd datasets/
ln -s /path/to/coco coco
```
Download the pretrained model in OneDrive or in the Baidu Cloud with code qr6o to train with the CSPDarkNet-53 backbone (optional)
```
mkdir pretrained_models
# download the `cspdarknet53.pth` to the `pretrained_models` directory
```
Train with yolof
```
python ./tools/train_net.py --num-gpus 8 --config-file ./configs/yolof_R_50_C5_1x.yaml
```
Test with yolof
```
python ./tools/train_net.py --num-gpus 8 --config-file ./configs/yolof_R_50_C5_1x.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
Note that there might be API changes in future detectron2 releases that make the code incompatible.
# 生成onnx
需要先拉取代码库里的detectron2
然后执行
```
python -m pip install -e detectron2
```

然后执行
```
python ~/detectron2/tools/deploy/export_model.py --format onnx --export-method caffe2_tracing --config-file ~/yoloF_test/YOLOF/configs/yolof_X_101_64x4d_C5_1x.yaml --output ./output3 MODEL.DEVICE cuda MODEL.WEIGHTS /root/yoloF_test/YOLOF/pretrained_models/YOLOF_X_101_64x4d_C5_1x.pth
```
这样会在output3 底下生成一个onnx文件

# 使用tensorrt


