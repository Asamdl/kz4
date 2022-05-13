import copy
from random import randint

import cv2
import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
from matplotlib import pyplot as plt
from numba import njit, prange
from scipy.ndimage import gaussian_filter

from scipy.signal import find_peaks


def getting_image_from_pixels(pixels: np.ndarray) -> Image:
    image = Image.new(size=(pixels.shape[0],pixels.shape[1]),mode="L")
    draw = ImageDraw(image)
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            draw.point((i, j), round(pixels[i][j]))
    return image


def getting_image_from_path(path: str) -> Image:
    image = Image.open(path)
    image = image.convert("L")
    return image


def getting_array_for_image(image: Image) -> np.ndarray:
    height = image.size[0]
    width = image.size[1]
    pix = image.load()
    pixels = np.ndarray(image.size)
    for i in range(height):
        for j in range(width):
            pixels[i][j] = pix[i, j]
    return pixels


@njit(parallel=True, fastmath=True)
def getting_diagram(pixels: np.ndarray) -> np.array:
    diag = np.zeros((256, 1))
    for j in prange(pixels.shape[1]):
        for i in prange(pixels.shape[0]):
            diag[round(pixels[i, j])][0] += 1
    return diag
@njit(parallel=True, fastmath=True)
def convert_array3_to_array1(pixels: np.ndarray):
    new_pixels =  np.zeros((pixels.shape[0],pixels.shape[1]))
    for i in prange(pixels.shape[0]):
        for j in prange(pixels.shape[1]):
            new_pixels[i][j] = round(pixels[i][j])
    return new_pixels.astype(np.uint8)
@njit(parallel=True, fastmath=True)
def sequential_approximation(pixels: np.ndarray)->np.ndarray:
    diag = getting_diagram(pixels)
    sum_p = 0
    threshold = 0
    delt = 1
    for i in range(256):
        sum_p += diag[i][0]
        if (sum_p / (pixels.shape[0] * pixels.shape[1])) > 0.5:
            threshold = i
            break
    while (delt != 0):
        sum_pix = 0
        count_pix = 0
        for i in prange(pixels.shape[0]):
            for j in prange(pixels.shape[1]):
                if (round(pixels[i][j]) in range(0, threshold)):
                    sum_pix += pixels[i][j]
                    count_pix += 1
        T1 = round(sum_pix / count_pix)
        sum_pix = 0
        count_pix = 0
        for i in prange(pixels.shape[0]):
            for j in prange(pixels.shape[1]):
                if (round(pixels[i][j]) in range(threshold, 255)):
                    sum_pix += pixels[i][j]
                    count_pix += 1
        T2 = round(sum_pix / count_pix)
        T_old = threshold
        threshold = round((T1 + T2) / 2)
        T_new = threshold
        delt = abs(T_new - T_old)

    for i in prange(pixels.shape[0]):
        for j in prange(pixels.shape[1]):
            if (pixels[i][j] >= threshold):
                pixels[i][j] = 255
            else:
                pixels[i][j] = 0
    return pixels
@njit(parallel=True, fastmath=True)
def P_tile(pixels: np.ndarray, P=0.5):
    diag = getting_diagram(pixels)
    sum_p = 1
    threshold = 0
    for i in prange(255):
        sum_p += diag[i][0]
        if (sum_p / (pixels.shape[0] * pixels.shape[1])) > P:
            threshold = i
            break
    for i in prange(pixels.shape[0]):
        for j in prange(pixels.shape[1]):
            if (pixels[i][j] >= threshold):
                pixels[i][j] = 255
            else:
                pixels[i][j] = 0

    return pixels

def get_image_from_path(path):
  image = getting_image_from_path(path)
  image = getting_array_for_image(image)
  return image
def get_image_from_pixels(pixels):
  return getting_image_from_pixels(pixels)

def average(mass):
    return sum(mass) / len(mass)

def median(mass):
    mass = sorted(mass)
    return mass[round(len(mass) / 2)]

def min_max(mass):
    return (min(mass) + max(mass)) / 2
@njit(parallel=True, fastmath=True)
def adaptive_threshold_average(pixels: np.ndarray, r, T):
    new_pixels = np.zeros((pixels.shape))
    for i in prange(pixels.shape[0]):
        for j in prange(pixels.shape[1]):
            mass = []
            for x in prange(-r, r):
                for y in prange(-r, r):
                    if i + x in range(0, pixels.shape[0] - 1) and j + y in range(0, pixels.shape[1] - 1):
                        mass.append(pixels[i + x][ j + y])
            if ((pixels[i][ j] - (sum(mass) / len(mass))) >= T):
                new_pixels[i][ j]= 255
            else:
                new_pixels[i][ j]= 0
    return new_pixels

@njit(parallel=True, fastmath=True)
def adaptive_threshold_median(pixels: np.ndarray, r, T):
    new_pixels = np.zeros((pixels.shape))
    for i in prange(pixels.shape[0]):
        for j in prange(pixels.shape[1]):
            mass = []
            for x in prange(-r, r):
                for y in prange(-r, r):
                    if i + x in range(0, pixels.shape[0] - 1) and j + y in range(0, pixels.shape[1] - 1):
                        mass.append(pixels[i + x][ j + y])
            mass = sorted(mass)
            if ((pixels[i][ j] - (mass[round(len(mass) / 2)])) >= T):
                new_pixels[i][ j]= 255
            else:
                new_pixels[i][ j]= 0
    return new_pixels
@njit(parallel=True, fastmath=True)
def adaptive_threshold_min_max(pixels: np.ndarray, r, T):
    new_pixels = np.zeros((pixels.shape))
    for i in prange(pixels.shape[0]):
        for j in prange(pixels.shape[1]):
            mass = []
            for x in prange(-r, r):
                for y in prange(-r, r):
                    if i + x in range(0, pixels.shape[0] - 1) and j + y in range(0, pixels.shape[1] - 1):
                        mass.append(pixels[i + x][ j + y])
            if ((pixels[i][ j] - ((min(mass) + max(mass)) / 2)) >= T):
                new_pixels[i][ j]= 255
            else:
                new_pixels[i][ j]= 0
    return new_pixels

def data_distribution(array, cluster, n, k):
    cluster_content = [[] for i in range(k)]

    for i in range(n):
        min_distance = 1000000.0
        situable_cluster = -1
        for j in range(k):
            distance = (array[i] - cluster[j]) ** 2

            distance = distance ** (1 / 2)
            if distance < min_distance:
                min_distance = distance
                situable_cluster = j

        cluster_content[situable_cluster].append(array[i])

    return cluster_content


def cluster_update(cluster, cluster_content):
    k = len(cluster)
    for i in range(k):
        updated_parameter = 0
        for j in range(len(cluster_content[i])):
            updated_parameter += cluster_content[i][j]
        if len(cluster_content[i]) != 0:
            updated_parameter = updated_parameter / len(cluster_content[i])
        cluster[i] = round(updated_parameter)
    return cluster


def clusterization(pixels: np.ndarray, k):
    count = 0
    array = np.array([0.0 for i in range(pixels.shape[0] * pixels.shape[1])])
    for i in prange(pixels.shape[0]):
        for j in prange(pixels.shape[1]):
            array[count] = pixels[i][j]
            count += 1

    n = len(array)

    cluster = [0.0 for q in range(k)]
    cluster_content = [[] for i in range(k)]


    for q in range(k):
        cluster[q] = randint(0, 255)

    cluster_content = data_distribution(array, cluster, n, k)
    privious_cluster = copy.deepcopy(cluster)
    while 1:
        cluster = cluster_update(cluster, cluster_content)
        cluster_content = data_distribution(array, cluster, n, k)
        if cluster == privious_cluster:
            break
        privious_cluster = copy.deepcopy(cluster)
    cluster = sorted(cluster)
    cluster_g = [0]
    for _k in range(k):
        cluster_g.append(cluster[_k])
    cluster_g.append(255)
    color_g = []
    for _k in range(k + 1):
        color_g.append(round(255 / (k + 1 - _k)))
    return (cluster_g,color_g)
@njit(parallel=True, fastmath=True)
def print_cluster(pixels,cluster,color):

    for i in prange(pixels.shape[0]):
        for j in prange(pixels.shape[1]):
            for _k in range(len(cluster) - 1):
                if (pixels[i][ j] in range(cluster[_k], cluster[_k + 1])):
                    pixels[i][ j]=  color[_k]
                    break

    return pixels


def canny_1(pixels):
  gray = pixels
  _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
  edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

  contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]
  result = np.zeros_like(pixels)
  for x in contours:
    cv2.drawContours(result, [x], 0, (255,255,255), cv2.FILLED)

  return result





def kmean(pixels,k):
    pixel_vals = pixels.reshape(-1, 3)
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(pixels.shape)

    return segmented_image

def histogram_parallel(pixels,size):
    kernel_size = size
    new_array = np.zeros((pixels.shape))
    for j in prange( pixels.shape[0]):
        for i in prange( pixels.shape[1] ):
            avg = np.round(np.mean(pixels[i:i + kernel_size, j:j + kernel_size]))
            new_array[j][i] = avg
    return new_array

@njit(fastmath=True, parallel=True)
def histogram_parallel_l(pixels, new_array):
    for j in prange(pixels.shape[0]):
        for i in prange(1, pixels.shape[1] - 1):
            avg = np.round((int(pixels[j][i - 1]) + int(pixels[j][i]) + int(pixels[j][i + 1])) / 3)
            new_array[j][i] = avg
    return new_array

def secondPeaks(pixels,size):
    new = histogram_parallel(pixels,size)
    ret = copy.deepcopy(new)
    #diag = getting_diagram(pixels)
    #count = 0
    #W = 12
    # for i in range(W,256-W):
    #     N = sum(diag[-W+i:W+i])
    #
    #     Peak =  1-(diag[i-W]+diag[i+W])/2*diag[i]*(1-N[0]/(2*W*diag[i]))
    #     if (Peak[0]<0):
    #         count+=1
    new_s = np.asarray(new).ravel()
    fig, ax = plt.subplots()
    ax.hist(new_s, 256, density=True, facecolor='b')
    new_peaks, _ = find_peaks(new_s, height=175)
    plt.title(f'Пиков {len(new_peaks)}')
    plt.show()
    return ret

@njit(fastmath=True, parallel=True)
def secondPeaks_l(img):
    container = copy.deepcopy(img)
    new = histogram_parallel_l(container, container)
    new_s = np.asarray(new).ravel()
    fig, ax = plt.subplots()
    ax.hist(new_s, 256, density=True, facecolor='b')
    plt.title('Еще раз сглаженная гистограмма')
    new_peaks, _ = find_peaks(new_s, height=125)
    print(str(new_peaks.size))
    plt.show()