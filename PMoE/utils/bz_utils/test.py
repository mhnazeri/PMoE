"""source: https://github.com/dotchen/LearningByCheating/blob/release-0.9.6/bird_view/utils/bz_utils/test.py"""
import video_maker as video_maker


import time
import numpy as np


tmp = np.zeros((256, 128, 3), dtype=np.uint8)
video_maker.init()


for i in range(256):
    tmp[:, :, 0] += 1
    video_maker.add(tmp)

    if i == 100:
        break
