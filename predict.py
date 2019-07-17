from keras.models import load_model
import pickle
import cv2
import numpy as np
from keras import backend as k
import random


class_colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(41)  ]


file=open("y_cat.pickle","rb")
y=pickle.load(file)
file.close()
y=np.array(y)
print(y.shape)
print(y[0][0][0].shape)


file=open("x_train.pickle","rb")
x=pickle.load(file)
file.close()

x=np.array(x)
x=np.moveaxis(x,1,3)
print(np.expand_dims(x[0],axis=0).shape)

model=load_model("model1.h5")
out=model.predict(np.expand_dims(x[0],axis=0))

image=np.zeros((128,128),float)
print(out[0][0][0].shape)
print(out[0][0].shape)
print(out[0].shape)
print(out.shape)


image=np.zeros((128,128,3),float)
indxe=0
for i in range(128):
    for j in range(128):
        a=out[0][i][j]
        max=-10000
        for k in range(41):
            if a[k]>max:
                max=a[k]
                index=k
#                print(k)
        image[i][j][0] = class_colors[index][0]
        image[i][j][1] = class_colors[index][1]
        image[i][j][2] = class_colors[index][2]
        print(out[0][i][j])


#print(image)
cv2.imshow('image',image)
cv2.waitKey(0)

#print(y[0][0][1])
#out=k.argmax(out,axis=3)

#print()