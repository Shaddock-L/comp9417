import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
import os 
import time


# reduce  the pictures dimension
# from 512 * 512 * 3 to 512 * 512 * 1
def reduce_to_1d(pic, w_r, w_g, w_b):
    r, g, b = pic[:, :, 0], pic[:, :, 1], pic[:, :, 2]
    res = w_r * r + w_g * g + w_b * b
    return res

# data augmentation (why and how are written in the report)
def transform():
    """
    https://pytorch.org/vision/stable/transforms.html
    """
    return transforms.Compose(
            [
            transforms.RandomResizedCrop(512, scale=(0.5, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.2),
            transforms.RandomAffine(degrees=(-15, 15), translate=(0.0, 0.5)),
            transforms.RandomHorizontalFlip(0.5), 
            transforms.RandomAutocontrast(0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            ]
        )

#plot the confusion matrix
def plot_cmat(cmat,classes):
    sns.heatmap(cmat.T, square=True, annot=True, cbar=False)          
    plt.xlabel('True label')     
    plt.ylabel('predicted label')
    plt.savefig('CNN_Confusion_Mat.png')
    plt.show()

#Our CNN Model (based on AlexNet)
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        #512 * 512
        self.conv1 = nn.Sequential(
            nn.Conv2d(              
            in_channels = 1,
            out_channels = 96,
            kernel_size = 11,
            stride = 4,
            padding = 1                
            ),
            #126 * 126 
            nn.ReLU(),
            nn.MaxPool2d(4, 2),  
            # 62*62
            nn.Conv2d(
            in_channels = 96,
            out_channels = 256,
            kernel_size = 5,
            stride = 1,
            padding = 2             
            ),        
            nn.ReLU(),
            #62*62
            nn.MaxPool2d(3, 2)
        )
        #30*30
        self.conv2 = nn.Sequential(
            nn.Conv2d(
            in_channels = 256,
            out_channels = 384,
            kernel_size = 3,
            stride = 1,
            padding = 1          
            ),
            #30 * 30                  
            nn.ReLU(),
            #120 * 120
            nn.Conv2d(
            in_channels = 384,
            out_channels = 384,
            kernel_size = 3,
            stride = 1,
            padding = 1  
            ),
            nn.ReLU(),
            #30 * 30
            nn.Conv2d(
            in_channels = 384,
            out_channels = 200,
            kernel_size = 3,
            stride = 1,
            padding = 1  
            ),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
        )
        #7 * 7

        self.out = nn.Sequential( 
            nn.Linear(7 * 7 * 200, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, 4)
            )
             
             
    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.conv2(x)
        x=x.view(-1, 7 * 7 * 200)
        output = self.out(x)
        #print('output size : ', output.size())
        return output


if __name__ == '__main__':
    file1 = 'X_train.npy'
    file2 = 'y_train.npy'
    testfile = 'X_test.npy'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    prev = time.time()

    #parameters
    epoch_num = 200
    train_size = 688
    test_size = 170
    weight_r = 0.05
    weight_g = 0.75
    weight_b = 0.2
    learning_rate =5e-5
    enhance_rate = 1.4

    net = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(net.parameters(), lr=learning_rate)

    #train part
    for epoch in range(epoch_num):
        total_correct = 0
        train_loss = 0
        
        X_train = np.load(file1,mmap_mode='r+')
        y_train = np.load(file2, mmap_mode='r+')
        trainloaderX = DataLoader(X_train)
        trainloadery = DataLoader(y_train)
        X_it = iter(trainloaderX)
        y_it = iter(trainloadery)

        for i in range(train_size):
            #read file from the memo
            try: 
                X_t = next(X_it)
                Y = next(y_it)
            except StopIteration:
                pass
            
            #reduce the size and enhance the features
            X = reduce_to_1d(X_t[0], weight_r, weight_g, weight_b)
            X = X.reshape(1, 1024, 1024)
            X = transform()(X)
            X = X.to(device)
            Y = Y.to(device)

            #Gradient zeroing
            optimiser.zero_grad()
            #predict
            outputs = net(X)

            #enhance the loss rate if type 3 is misclassified
            if Y.item() == 3 and outputs.argmax(dim=1).item() != 3:
                loss = enhance_rate * criterion(outputs,Y.long())
            else:
                loss = criterion(outputs,Y.long())
            loss.backward()
            #use gradiant to optimize our net
            optimiser.step()
            train_loss += loss.item()

            #get the predict value
            output = outputs.argmax(dim=1)
            if(epoch % 20 == 0):
                print('y is: ', Y.item(), 'ouput is: ', output)

        print(epoch+1,'  ', train_loss)

    torch.save(net.state_dict(), 'savedModel.pth')
    print("Model saved to savedModel.pth")

    #test part
    correct = 0
    total = 0
    pred_list = []
    true_list = []
    with torch.no_grad():
        for i in range(test_size):
            try:
                X_tes = next(X_it)
                Y_test = next(y_it)
            except StopIteration:
                pass
            X_test = reduce_to_1d(X_tes[0], weight_r, weight_g, weight_b)
            X_test = X_test.reshape(1, 1024, 1024)
            X_test = transform()(X_test)
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)

            #output
            outputs = net(X_test) 
            _, predicted = torch.max(outputs.data, dim=1)  

            total += Y_test.size(0)
            pred_list.append(outputs.argmax(dim=1).item())
            true_list.append(Y_test.item())

            correct += (predicted == Y_test).sum()  # same = 1，diff = 0
            print('y is: ',Y_test.item() , '     predict is : ' ,predicted) 
        print("corrtecness：{:.3f}".format(torch.true_divide(correct,total)))
    y_pred = np.array(pred_list)
    y_true = np.array(true_list)


    #result part
    X_ret = np.load(testfile,mmap_mode='r+')
    testloaderX = DataLoader(X_ret)
    X_ret_it = iter(testloaderX)
    save_list = []
    with torch.no_grad():
        for i in range(len(X_ret)):
            try:
                X_tes = next(X_ret_it)
            except StopIteration:
                pass
            X_test = reduce_to_1d(X_tes[0], weight_r, weight_g, weight_b)
            X_test = X_test.reshape(1, 1024, 1024)
            X_test = transform()(X_test)
            X_test = X_test.to(device)

            #output
            outputs = net(X_test)
            save_list.append(outputs.argmax(dim=1).item())
    
    to_save = np.array(save_list).astype(np.float64)
    np.save('y_test_of_CNN_(Best_Performance).npy', to_save)
    
    #draw the confusion matrix
    confusion_mtx = confusion_matrix(y_true, y_pred)
    label_map = {0: 'type0', 1: 'type1', 2: 'type2', 3: 'type3'}
    plot_cmat(confusion_mtx,classes = list(label_map.values()))

    print('f1(weighted) score is : ',f1_score(true_list, pred_list, average='weighted'))
    print('f1(macro) score is : ',f1_score(true_list, pred_list, average='macro'))    
    now = time.time()
    #print(f'use {now-prev} s')

    






