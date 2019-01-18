import h5py
import numpy as np
import cv2 as cv
import PIL.Image as Image
import math

filename = 'results.h5'
h = h5py.File(filename,'r')

Save_PATH_AFFIMG = r'./affimg/'

res = h['results']
resori = np.array(res)
resori = resori[0,1,:,:]
resnew = 255*cv.resize(resori,(512,424),interpolation=cv.INTER_CUBIC)

#affimg = Image.fromarray(resnew.astype(np.float32),mode='F')
#affimg = affimg.convert('RGB')
affimg_path = Save_PATH_AFFIMG + 'affimg.png'
cv.imwrite(affimg_path,resnew)
#affimg.save(affimg_path)
print("Succeed to save affordance images")
