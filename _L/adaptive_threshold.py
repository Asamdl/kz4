import copy

from matplotlib import pyplot as plt

from getting import get_image_from_path, get_image_from_pixels, P_tile, sequential_approximation, \
    adaptive_threshold_min_max, kmean, adaptive_threshold_median, adaptive_threshold_average

path = "2015_01054.jpg"
image = get_image_from_path(path)
im1 = copy.deepcopy(image)
im2 = copy.deepcopy(image)
im3 = copy.deepcopy(image)
pic_box = plt.figure(figsize=(16,4))
pic_box.add_subplot(1,3,1)
T = -2
R = 3
plt.imshow(get_image_from_pixels(adaptive_threshold_min_max(im1,R,T)),cmap='gray')
plt.title(f"average R,T = {R},{T}")
plt.axis('off')
pic_box.add_subplot(1,3,2)
K = 6
plt.imshow(get_image_from_pixels(adaptive_threshold_median(im2,R,T)),cmap='gray')
plt.title(f"median R,T = {R},{T}")
plt.axis('off')
pic_box.add_subplot(1,3,3)
K = 8
plt.imshow(get_image_from_pixels(adaptive_threshold_average(im3,R,T)),cmap='gray')
plt.title(f"average R,T = {R},{T}")
plt.axis('off')
plt.show()