# 分别训练title 和 image 的顶层脚本
#  模型参数在相应的model的config.py中调整
#  训练参数在相应的train的config.py中调整

import train.image_pre_train.train as image_pretrain
import train.title_pre_train.train as title_pretrain

title_pretrain.pretrain()
print("image pretrain")
image_pretrain.pretrain()
