import copy

from matplotlib import pyplot as plt
import numpy as np
from getting import get_image_from_path, get_image_from_pixels, P_tile, sequential_approximation, kmean, clusterization, \
    print_cluster

path = "2015_01054.jpg"
image = get_image_from_path(path)
im1 = copy.deepcopy(image).astype(np.uint8)
im2 = copy.deepcopy(image)
pic_box = plt.figure(figsize=(10,4))
pic_box.add_subplot(1,2,1)
K = 2
param = clusterization(im1,K)
plt.imshow(get_image_from_pixels(print_cluster(im1,param[0],param[1])),cmap='gray')
plt.title(f"kmean K = {K}")
plt.axis('off')
K = 4
pic_box.add_subplot(1,2,2)
param = clusterization(im2,K)
plt.imshow(get_image_from_pixels(print_cluster(im2,param[0],param[1])),cmap='gray')
plt.title(f"kmean K = {K}")
plt.axis('off')
plt.show()