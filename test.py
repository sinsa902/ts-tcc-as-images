import numpy as np
import cv2
import os

rgb_image1 = cv2.imread(os.path.join("server_1.png"))
rgb_image2 = cv2.imread(os.path.join("local_1.png"))
rgb_image1 = cv2.cvtColor(rgb_image1, cv2.COLOR_BGR2RGB)
rgb_image2 = cv2.cvtColor(rgb_image2, cv2.COLOR_BGR2RGB)

rgb_image1 = cv2.resize(rgb_image1, (224,224))
rgb_image2 = cv2.resize(rgb_image2, (224,224))

rgb_image1 = np.transpose(rgb_image1, (2, 0, 1))
rgb_image2 = np.transpose(rgb_image2, (2, 0, 1))

