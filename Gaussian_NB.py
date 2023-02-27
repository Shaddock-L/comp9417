import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
# a function to select data from .npy file
# the purpose of the function is change the '1024x1024' pixels into 'sp x sp' pixels
# X_file is the name of the file
# n is the number of the objects
# sp is the sqrt number of parts that we split the picture 
def creat_dataset(X_file,sp=64):
    X_raw=np.load(X_file,mmap_mode='r+')
    dl=DataLoader(X_raw)
    d=iter(dl)
    X=[]
    
    while True:
        try:
            t=np.array(next(d))[0]
            l=[]
            t1=np.split(t,sp)
            for i in t1:
                t2=np.split(i,sp,1)
                for j in t2:
                    r=j[:,:,0].mean()
                    g=j[:,:,1].mean()
                    b=j[:,:,2].mean()
                    l+=[0.3*r+0.3*g+0.3*b]
            X+=[l]
        except:
            break
    X=np.array(X).round(1)
    return X
# load y_train file
y=np.load('y_train.npy')
n=len(y)
# apply the above function
X=creat_dataset('X_train.npy',64)
# split the dataset into train and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=42)
# creat a gaussian naive bayes models
clf = GaussianNB()
# fit the dataset
clf.fit(X_train,y_train)
# predict
y_prd=clf.predict(X_test)
# calculate and print accuracy
a=accuracy_score(y_test,y_prd)
print(f'accuracy_score: {a}')
f1=f1_score(y_test,y_prd,average='weighted')
print(f'f1_score: {f1}')
#predict the data in X_test.npy
x_test=creat_dataset('X_test.npy',64)
pred=clf.predict(x_test)
print(pred)
#save the file as .npy
np.save('y_test_of_GaussianNB.npy',pred)