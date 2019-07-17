import os
import pickle
import numpy as np
import cv2

i=0
image_num=0
x=[]
x_final=[]
x_val=[]
Dir="C:\\Users\\ANKIT\\Desktop\\dataset\\input"
for image_name in os.listdir(Dir):
    print(image_name)
    image_path=os.path.join(Dir,image_name)

    image=cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
    image = image.astype('float32')
    image=image/255.0
    print(image.dtype)
    #image=np.array(image)

    x.append(image)
    i=i+1
    if i==7 and image_num<1500:
        #x_final.append(x)
        i=0
        temp=np.array(x)
        print(temp.shape)
        x_final.append(temp)
        #x_final=np.array(x_final)
        #print(temp.shape)
        del x[0:7]
        image_num=image_num+1
    elif i==7 and image_num >=1500:
        i = 0
        temp = np.array(x)
        print(temp.shape)
        x_val.append(temp)
        del x[0:7]
        image_num = image_num + 1
#print(x_final)
file=open("x_train.pickle","wb")
pickle.dump(x_final,file)
file.close()

file1=open("x_val.pickle","wb")
pickle.dump(x_val,file1)
file1.close()
print(np.array(x_val).shape)
print(np.array(x_final).shape)