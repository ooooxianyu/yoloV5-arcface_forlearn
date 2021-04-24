# yoloV5-arcface_forlearn

```
     ███████╗ ██████╗ ████████╗ ██╗   ██╗
     ██╔════╝   ██╔═╝ ██╔═════╝ ██║   ██║
     █████║     ██║   ████████║ ████████║
     ██╔══╝     ██║   ╚════╗██║ ██╔═══██║
     ██║      ██████╗ ████████║ ██║   ██║
     ╚═╝      ╚═════╝ ╚═══════╝ ╚═╝   ╚═╝
```
## 之前学习人脸识别整合的一系列源码：仓库如下。
for learning...
yoloV5官方源码：https://github.com/ultralytics/yolov5 
arcFace源码： https://github.com/ronghuaiyang/arcface-pytorch
silentFace静默活体检测：https://github.com/smisthzhu/Silent-Face-Anti-Spoofing

另外有兴趣的可以去我博客看看：https://blog.csdn.net/weixin_41809530/article/details/107313752

# 人脸识别实战模型

## 1.模型介绍
    --人脸检测：yoloV5对人脸进行检测
    --人脸识别：arcface
    --活体检测：silentFace静默活体检测

## 2.预训练模型
    --weights 权重包的下载地址：包含yoloV5的预训练权重包，训练好的侦测人脸的权重包best，人脸识别的权重包resnet110 链接：https://pan.baidu.com/s/1YzgQcFVl4Rd6skN5q7mw-w 提取码：kusi

## 3.数据集
    --人脸检测：celebA
    --人脸识别：CASIA-WebFace （我没有自己训练，直接用arcface源码提供的权重）

## 4.运行
    --修改main内的参数配置，后直接运行main.py。
    --展示效果可以参见我的博客。
    
## 5.不足
    --由于都是用开源数据集，以及各源码拼接的，所以仅供学习参考。
    --网络性能和处理都没有特别的优化，因此开启活体检测帧数都会比较低。
    --活体检测检测的效果因摄像头型号和使用场景鲁棒性受限，实际效果自己看，我设置只有预测为真脸且预测率大于0.9才判定为真，否则都为假。
    
## 6.TODO
    --可能会写各requirement.txt吧，暂时只能跑main的时候缺啥下啥。
    --后续可能完善人脸检测，加入各人脸检测模型选项。如MTCNN、RetinaFace……
    --可能加入人脸关键点部分代码，实现眨眼等人脸动作识别。
    --暂时没什么时间所以不要抱太大希望，可以自己实现看看 hhhhhh