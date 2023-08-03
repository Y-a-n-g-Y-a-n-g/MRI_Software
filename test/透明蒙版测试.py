# -*- coding: utf-8 -*-
"""
@Time ： 12/24/2022 1:24 PM
@Auth ： YY
@File ：透明蒙版测试.py
@IDE ：PyCharm
@state:
@Function：
"""
import numpy as np
import os
import cv2

def put_mask(img_path,output_fold):


    image = cv2.imread(img_path)
    bbox1 = [72,41,208,330]

    # 3.画出mask
    zeros1 = np.zeros((image.shape), dtype=np.uint8)
    print(image.shape)
    zeros_mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    color=(0,255,0), thickness=-1 ) #thickness=-1 表示矩形框内颜色填充


    zeros_mask = np.array(zeros_mask1 )

    try:
        # alpha 为第一张图片的透明度
        alpha = 1
        # beta 为第二张图片的透明度
        beta = 0.2
        gamma = 0
        # cv2.addWeighted 将原始图片与 mask 融合
        mask_img = cv2.addWeighted(image, alpha, zeros_mask, beta, gamma)
        cv2.imwrite(os.path.join(output_fold,'mask_img.jpg'), mask_img)
    except:
        print('异常')

put_mask(img_path = 'temp3.png', output_fold='./')