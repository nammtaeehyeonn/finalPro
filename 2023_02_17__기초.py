# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # json 파일 다루기

import json

with open('E:\\다양한 형태의 한글 문자 OCR\\Training\\[라벨]Training_필기체\\1.글자\\002\\00230001001.json', encoding = "utf8") as f:
    json_object = json.load(f)

# 사진
json_object['image']['file_name']

# 라벨
json_object['info']['text']

# 위 방법으로 라벨들을 하나의 tensor 형태로 저장해서 사용하면 될 듯.
#
# 코랩으로 돌리는게 나을 듯

# # 이미지 <-> Numpy

import numpy as np
from PIL import Image
import torch

img = Image.open('E:\\다양한 형태의 한글 문자 OCR\\Training\\[원천]Training_필기체\\1.글자\\001\\00130001001.jpg')
img.show()

x = np.array(img)  # 이미지 -> 넘파이
print(x)
print(x.shape)   # (100,111,3)

img_2 = Image.fromarray(x) # 넘파이 -> 이미지
img_2.show()

# ![image.png](attachment:image.png)

y = x.copy()

y[:,:,0]   # 기존 3차원 데이터를 3개 차원으로 나눌수 있음.

y[:,:,0].shape

img_3 = Image.fromarray(y[:,:,0]) # NumPy array to PIL image
img_3.show()

# ![image.png](attachment:image.png)

img_3 = Image.fromarray(y[:,:,1]) # NumPy array to PIL image
img_3`.show()

# ![image.png](attachment:image.png)

img_3 = Image.fromarray(y[:,:,2]) # NumPy array to PIL image
img_3.show()

# ![image.png](attachment:image.png)

# ### 3개 차원의 글씨가 별 반 다를게 없어보임 => 학습 데이터 증가 가능
# #### 약간 회전 하는 방식? => batch norm (어차피 모델에 포함 되어있음.)

scaled_y = y.copy()

scaled_y0 = scaled_y[:,:,0]
scaled_y1 = scaled_y[:,:,1]
scaled_y2 = scaled_y[:,:,2]

scaled_y0[(scaled_y0 >= 220)] = 255
scaled_y1[(scaled_y1 >= 220)] = 255
scaled_y2[(scaled_y2 >= 220)] = 255

scaled_y0[(scaled_y0 < 220)] = 0
scaled_y1[(scaled_y1 < 220)] = 0
scaled_y2[(scaled_y2 < 220)] = 0

# #### 여기서 위 기준을 220이 아니라 240, 230, 220 으로 나누면 약간의 노이즈 차이 발생
# => 오버피팅 방지 효과도 있을 듯

img_3 = Image.fromarray(scaled_y0) # NumPy array to PIL image
img_3.show()

# ![image.png](attachment:image.png)

img_3 = Image.fromarray(scaled_y1) # NumPy array to PIL image
img_3.show()

# ![image.png](attachment:image.png)

img_3 = Image.fromarray(scaled_y2) # NumPy array to PIL image
img_3.show()

# ![image.png](attachment:image.png)


