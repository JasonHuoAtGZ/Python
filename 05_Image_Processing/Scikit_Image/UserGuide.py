import numpy as np
import skimage
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize



# import my own image
mypic=io.imread("C:/Users/jason/Pictures/Camera Roll/WIN_20171130_16_55_47_Pro.jpg")

#io.imshow(mypic)
#io.show()

# data type of imported image
print("Type: "+'\n'+str(type(mypic)))
print("Dimension: "+'\n'+str(mypic.shape))
print("1st Dimension: "+'\n'+str(mypic.shape[0]))
print("2nd Dimension: "+'\n'+str(mypic.shape[1]))
print("3rd Dimension: "+'\n'+str(mypic.shape[2]))

nrows, ncols, nchl = mypic.shape
row, col = np.ogrid[:nrows, :ncols]
cnt_row, cnt_col = nrows / 2, ncols / 2
outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 >(nrows / 2)**2)
mypic[outer_disk_mask] = 0

#io.imshow(mypic)
#io.show()

mypic_resized=resize(mypic, output_shape=[600,600,3])
io.imshow(mypic_resized)
io.show()


