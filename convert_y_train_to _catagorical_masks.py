import pickle
import numpy as np
from keras.utils import to_categorical


classes=41
file=open("y_train.pickle","rb")
y=pickle.load(file)
file.close()

file=open("y_val.pickle","rb")
y_val=pickle.load(file)
file.close()
y_val_cat=[]
y_cat=[]

for i in y_val:
    data=to_categorical(i,num_classes=classes,dtype='float32')
    y_val_cat.append(data)


file1=open("y_val_cat.pickle","wb")
pickle.dump(y_val_cat,file1)
file.close()
y_val_cat=np.array(y_val_cat)
print(y_val_cat.shape)

for i in y:
    data=to_categorical(i,num_classes=classes,dtype='float32')
    y_cat.append(data)


file1=open("y_cat.pickle","wb")
pickle.dump(y_cat,file1)
file.close()
y_cat=np.array(y_cat)
print(y_cat.shape)
#print(y_cat)