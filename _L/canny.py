import copy

from matplotlib import pyplot as plt
import numpy as np
from getting import get_image_from_path, get_image_from_pixels, P_tile, sequential_approximation, canny_1

path = "2015_01054.jpg"
image = get_image_from_path(path)
im1 = copy.deepcopy(image)
im2 = copy.deepcopy(image)
pic_box = plt.figure(figsize=(10,4))
pic_box.add_subplot(1,1,1)
c = canny_1(im1.astype(np.uint8))
r = get_image_from_pixels(c)
plt.imshow(r,cmap='gray')
plt.title(f"Поиск неоднородностей ")
plt.axis('off')
plt.show()