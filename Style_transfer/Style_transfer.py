'''
基于Tensorflow2，TF-Hub开源项目——神经网络风格迁移
'''

# 导入和配置模块
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import time

# 张量转化为图像，保存图片
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  # 保存图片
  img = tf.image.encode_png(tensor)
  with tf.io.gfile.GFile("./Style_transfer_Output.png", 'wb') as file:
    file.write(img.numpy())
  return PIL.Image.fromarray(tensor)


# 定义一个加载图像的函数，并将其最大尺寸限制为 512 像素
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

# 创建一个简单的函数来显示图像
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)
  plt.imshow(image)
  if title:
    plt.title(title)


content_image = load_img("./test_picture/girl.jpg")
style_image = load_img("./test_picture/MonaLisa.jpg")

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')
plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

# 使用 TF-Hub 进行快速风格迁移
hub_module = hub.load('./model/magenta_arbitrary-image-stylization-v1-256_1')
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)
