import copy

from matplotlib import pyplot as plt

from getting import get_image_from_path, get_image_from_pixels, P_tile, sequential_approximation

path = "2015_01054.jpg"
image = get_image_from_path(path)
im1 = copy.deepcopy(image)
pic_box = plt.figure(figsize=(10,4))
pic_box.add_subplot(1,1,1)
P = 0.3
plt.imshow(get_image_from_pixels(sequential_approximation(im1)),cmap='gray')
plt.title(f"sequential_approximation")
plt.axis('off')
plt.show()