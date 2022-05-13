import copy

from matplotlib import pyplot as plt

from getting import get_image_from_path, get_image_from_pixels, P_tile, sequential_approximation

path = "2015_01054.jpg"
image = get_image_from_path(path)
im1 = copy.deepcopy(image)
im2 = copy.deepcopy(image)
pic_box = plt.figure(figsize=(10,4))
pic_box.add_subplot(1,2,1)
P = 0.3
plt.imshow(get_image_from_pixels(P_tile(im1,P)),cmap='gray')
plt.title(f"P_tile P = {P}")
plt.axis('off')
P=0.6
pic_box.add_subplot(1,2,2)
plt.imshow(get_image_from_pixels(P_tile(im2,P)),cmap='gray')
plt.title(f"P_tile P = {P}")
plt.axis('off')
plt.show()