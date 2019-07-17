import pickle
import cv2
import os
import numpy as np
y=[]
y_val=[]
i=0
image_num=0
Dir="C:\\Users\\ANKIT\\Desktop\\dataset\\target"
for image_name in os.listdir(Dir):
    print(image_name)
    image_path=os.path.join(Dir,image_name)
    image=cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
    image=np.array(image)
    image = image.astype('float32')/255.0
    image=np.expand_dims(image,axis=2)

    #print(image)
    if image_num <1500:
        #print(image.shape)
        y.append(image)
        image_num = image_num + 1
        i=i+1
    elif image_num >=1500:
        y_val.append(image)
        image_num=image_num+1
        i=i+1
file=open("y_train.pickle","wb")
pickle.dump(y,file)
file.close()

file1=open("y_val.pickle","wb")
pickle.dump(y_val,file1)
file1.close()
#print(i)
y=np.array(y)
print(y.shape)
print(np.array(y_val).shape)

