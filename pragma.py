import cv2
import numpy as np

from getting import getting_diagram, getting_array_for_image, getting_image_from_path, sequential_approximation, \
    getting_image_from_pixels, convert_array3_to_array1, P_tile, average, median, min_max, \
    adaptive_threshold_average, adaptive_threshold_median, adaptive_threshold_min_max, clusterization, print_cluster, \
     canny_1, secondPeaks, kmean

# path = "L.png"
# image = getting_image_from_path(path=path)
# pixels = getting_array_for_image(image=image)
# diag = getting_diagram(pixels=pixels)
# new_im = sequential_approximation(pixels,diag)
# getting_image_from_pixels(new_im).show()

cap = cv2.VideoCapture("1.mp4")

while True:
    ret, frame = cap.read()

    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #--2.2--
        #frame = sequential_approximation(frame)

        #--2.1--
        #frame = P_tile(frame,P=0.8)

        #--3--
        #frame = adaptive_threshold_average(frame,3,0)
        #frame = adaptive_threshold_median(frame,1,7)
        #frame = adaptive_threshold_min_max(frame,5,0)\

        #--2.3--
        #frame = kmean(frame,4)

        #--1--
        #frame = canny_1(frame)


    except:
        cap.release()
        raise
    cv2.imshow('video feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()