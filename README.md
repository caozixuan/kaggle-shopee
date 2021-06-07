# kaggle-shopee
This is the training code for kaggle competition.
This code is for DLNN course project.

# simple description
## 方案一：分别单独训练image_encoder和title_encoder

### 设置

image_encoder:	resnet18并且只训练最后一层，冻结中间的其他层。

title_encoder:	两层的MLP，激活函数为LeakRelu()

loss: 
$$
loss_{i} = -log(\sigma (z_{i}^{T}z_{pos})) - \sum_{neg} log(\sigma (-z_{i}^{T}z_{neg}))
$$

### image_encoder的训练方式

对于每个图片，取1个正样本（同一组中随机取）和5个负样本（所有样本中随机取）。

将正负样本分别输入image_encoder得到新的特征表示,按照损失函数计算损失，更新image_encoder。

### title_encoder的训练方式

首先利用tfidf对所有的title进行语义的表示，由于这个表示为度太高，之后利用pca进行降维，降到128维作为title的源特征表示。

对于每个title信息，我们取1个正样本（同一组的title中随机取）和5个负样本（所有title中随机取）。

将正负样本分别输入title_encoder得到新的特征表示,按照损失函数计算损失，更新title_encoder。



## 方案二：训练一个encoder

### 设置

image_encoder:	resnet18并且只训练最后一层，冻结中间的其他层。

title_encoder:	两层的MLP，激活函数为 LeakRelu()

loss: 
$$
loss_{i} = -log(\sigma (z_{i}^{T}z_{pos})) - \sum_{neg} log(\sigma (-z_{i}^{T}z_{neg}))
$$

### encoder的训练方式

同方案一，只是将image_encoder和title_encoder的输出拼接起来计算loss。



## 方案三：efficient net + arc margin loss

### 设置

image_encoder： efficient net 4

loss：arc margin + cross entropy loss

### 训练方式

按照数据的分组方式对数据进行打标签处理。

对每张图片，输入到efficient net 4 提取特征表示，对特征表示旋转一定的角度之后（即加入arc margin）计算交叉熵损失，更新efficient net 4。



## 方案四：Sbert + arc margin loss

### 设置

title_encoder: 预训练的transformer

loss: arc margin + cross entropy loss

### 训练方式

按照数据的分组方式对数据进行打标签处理。

对每个title，输入到transformer提取特征表示，对特征表示旋转一定的角度之后（即加入arc margin）计算交叉熵损失，更新transformer。



# 方案五：efficient net + transformer + arc margin loss

### 设置

title_encoder:  预训练的transformer

image_encoder： efficient net 4

loss： arc margin + cross entropy loss

### 训练方式

按照数据的分组方式对数据进行打标签处理。

对每个（image，title），输入到efficient net和transformer提取特征表示并拼接起来，对拼接之后的特征表示旋转一定的角度之后（即加入arc margin）计算交叉熵损失，更新模型。

# 提交结果

unsupervised_baseline.ipynb: 最简单的不需要训练的baseline，下载预训练的resnet分类模型+tf-idf文本，最终测试集准确率0.709
method.ipynb: image最终使用了一个tensorflow训练结果:EfficientNetB3_512_42.h5 和两个pytorch模型，模型名称为tf_efficientnet_b5_ns和arcface_512x512_tf_efficientnet_b4_LR
CFG class下还有其他的模型使用，限于提交的运行时间限制，只能加载有限的模型，结果差不多，最终测试集准确率可以达到0.722.

使用的依赖和预训练模型在 https://pan.baidu.com/s/1DKvTF4OXMe26YD1LEMAsQw 密码 9c45
将文件解压后，里面全部文件和文件夹复制到当前文件夹下