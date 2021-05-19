# 【TF2开源项目】神经网络之风格迁移V2
## 前言
风格迁移，基于A图像内容，参考B图像的风格（名画，像毕加索或梵高一样绘画），创造出一幅新图像。

本文基于TF-Hub开源项目进行开发，60多行代码快速实现神经网络的风格迁移，为方便大家使用，已经整理相关代码和模型到Github中，直接下载即可使用。

## 原理
风格迁移是基于生成对抗网络实现的，是一种优化技术，用于将两个图像，A图像内容和B图像风格，混合再一起，是输出的图像看起来像A图像，但是也参考了B图像的风格。

通过优化输出图像，以匹配A图像的内容统计数和B图像的风格统计数据。这些统计数据可以使用卷积网络从图像中提取。

## 模型效果
<img src="https://github.com/guo-pu/Tensorflow2/blob/master/Style_transfer_V2/test_picture/Model_effect_v2.png" /><br/>

## 项目实践
**3.1、下载项目**

大家点击这里Github下载代码和模型

**3.2、运行环境**

主要用到几个依赖库：Tensorflow2.x、tensorflow_hub、numpy、PIL、matplotlib

**3.3、运行模型**

首先选择内容图像，content_image；风格图像，style_image；填写对应的图像路径。

然后直接运行代码 Style_transfer.py，运行成功后生成Style_transfer_Output.png。


