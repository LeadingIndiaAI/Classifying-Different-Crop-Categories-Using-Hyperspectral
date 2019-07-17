from model import *
import pickle
import numpy as np

# number of classes
classes=41

# loading x_train
file=open("x_train.pickle","rb")
x_train=pickle.load(file)
file.close()

# loading y_train

file=open("Y_TRAIN_CAT.pickle","rb")
y_cat=pickle.load(file)
file.close()
y_cat=np.array(y_cat)

# loading x_validation
file=open("x_val.pickle","rb")
x_val=pickle.load(file)
file.close()


# loading y_validation
file=open("Y_VAL.pickle","rb")
y_val=pickle.load(file)
file.close()
y_val=np.array(y_val)

# changing dimensions of x_train

x_train=np.array(x_train)
x_train=np.moveaxis(x_train,1,3)

x_val=np.array(x_val)
x_val=np.moveaxis(x_val,1,3)

model=unet(classes)
model.fit(
    x=x_train,
    y=y_cat,
    batch_size=16,
    epochs=50,
    validation_data=(x_val, y_val),
   )
model.save("model1.h5")