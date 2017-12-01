import numpy as np
import skimage
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from math import *

# import my own image
mypic=io.imread("C:/Users/jason/Pictures/Amazing Lock Screen/PupsPlayGalapagos_ZH-CN8090325795_1920x1200.jpg")

# data type of imported image
print("Type: "+'\n'+str(type(mypic)))
print("Dimension: "+'\n'+str(mypic.shape))
print("1st Dimension: "+'\n'+str(mypic.shape[0]))
print("2nd Dimension: "+'\n'+str(mypic.shape[1]))
print("3rd Dimension: "+'\n'+str(mypic.shape[2]))

def image_sliding_sampling(arr_image, int_win_row, int_win_col, int_sliding_row, int_sliding_col, str_save_path, str_save_name):
    image_count=1
    int_org_row, int_org_col, chl = arr_image.shape

    row_step=floor((int_org_row-int_win_row)/int_sliding_row)
    col_step=floor((int_org_col-int_win_col)/int_sliding_col)

    print("int_org_row="+str(int_org_row))
    print("int_org_col="+str(int_org_col))
    print("row_step="+str(row_step))
    print("col_step="+str(col_step))

    for i in range(int(row_step+1)):
        for j in range(int(col_step+1)):
            arr_image_sliced=arr_image[(i*int_sliding_row):(i*int_sliding_row+int_win_row-1),\
                                       (j*int_sliding_col):(j*int_sliding_col+int_win_col-1),:]
            #io.imshow(arr_image_sliced)
            #io.show()

            io.imsave(str_save_path+str_save_name+"_"+str(image_count)+".jpg",arr_image_sliced)

            '''
            print("===================================")
            print("This is the # "+str(image_count)+" image")
            print(i*int_sliding_row)
            print(i*int_sliding_row+int_win_row-1)
            print(j*int_sliding_col)
            print(j*int_sliding_col+int_win_col-1)
            '''

            image_count=image_count+1
    return

image_sliding_sampling(mypic, int_win_row=200, int_win_col=200, int_sliding_row=200, int_sliding_col=200, \
                       str_save_path="D:/99_Others/Image_Sampling/", \
                       str_save_name="sliced")