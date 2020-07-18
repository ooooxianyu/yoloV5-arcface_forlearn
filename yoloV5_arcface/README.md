结合yoloV5官方源码：https://github.com/ultralytics/yolov5
以及arcFace源码： https://github.com/ronghuaiyang/arcface-pytorch

CSDN：https://blog.csdn.net/weixin_41809530/article/details/107313752

weights 权重包的下载地址：包含yoloV5的预训练权重包，训练好的侦测人脸的权重包best，人脸识别的权重包resnet110
链接：https://pan.baidu.com/s/1YzgQcFVl4Rd6skN5q7mw-w 
提取码：kusi

========================================
更新：silentFace静默活体检测：https://github.com/smisthzhu/Silent-Face-Anti-Spoofing

权重包比较小，直接放代码里面。

具体效果展示也更新在博客里面。

运行的时候通过detect中open_rf控制是否开启真假脸检测。

但是运行的速度变低，检测的效果因摄像头型号和使用场景鲁棒性受限，实际效果自己看，我设置只有预测为真脸且预测率大于0.9才判定为真，否则都为假。