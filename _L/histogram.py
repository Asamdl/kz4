import copy

from matplotlib import pyplot as plt

from getting import get_image_from_path, get_image_from_pixels, P_tile, sequential_approximation, secondPeaks, \
    secondPeaks_l

path = "2015_01054.jpg"
image = get_image_from_path(path)

secondPeaks(image,2)
secondPeaks(image,3)
secondPeaks(image,4)
