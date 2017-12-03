import numpy as np
import skimage
import skimage.io as io
import matplotlib.pyplot as plt
import datetime as dt
import time

from skimage import data
from skimage.transform import resize
from math import *


def image_sliding_sampling(int_win_row, int_win_col, int_sliding_row, int_sliding_col, str_input_image, str_save_path, str_save_name):
    # pgm start time
    start = dt.datetime.today()
    # import image
    arr_image= io.imread(str_input_image)

    # data type of imported image
    print("===================================")
    print("Import Image Type: " + '\n' + str(type(arr_image)))
    print("Import Image Dimension: " + '\n' + str(arr_image.shape))

    # set sliding steps
    image_count=1

    # get image shape dimension
    int_org_row, int_org_col, chl = arr_image.shape

    # get sliding steps at rows & columns
    row_step=floor((int_org_row-int_win_row)/int_sliding_row)
    col_step=floor((int_org_col-int_win_col)/int_sliding_col)

    # start sliding
    for i in range(int(row_step+1)):
        for j in range(int(col_step+1)):
            arr_image_sliced=arr_image[(i*int_sliding_row):(i*int_sliding_row+int_win_row-1),\
                                       (j*int_sliding_col):(j*int_sliding_col+int_win_col-1),0] # gray-scaled only is saved
            #io.imshow(arr_image_sliced)
            #io.show()

            io.imsave(str_save_path+str_save_name+"_"+str(image_count)+".jpg",arr_image_sliced)

            image_count=image_count+1

    # pgm end time
    end = dt.datetime.today()

    # print pgm runtime
    print("Program Runtime: "+str(end-start))
    return

image_sliding_sampling(int_win_row=80, int_win_col=60, int_sliding_row=80, int_sliding_col=60, \
                       str_input_image="D:/99_Others/Image_Sampling/Face_Detection/raw/Murder_On_the_Orient_Express_2017/093632.61539587_1000X1000.jpg" ,\
                       str_save_path="D:/99_Others/Image_Sampling/Face_Detection/raw/Murder_On_the_Orient_Express_2017/sliced_01/", \
                       str_save_name="sliced")
