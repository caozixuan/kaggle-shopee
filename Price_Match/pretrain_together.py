# title 和 image 一起训练的顶层脚本
#  模型参数在model的config.py中调整
#  训练参数在相应的train的config.py中调整

from train.pre_train.train import pretrain

pretrain()
