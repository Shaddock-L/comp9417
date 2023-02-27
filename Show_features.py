import matplotlib.pyplot as plt
import numpy as np

X_train = np.load('X_train.npy', mmap_mode='r+')
X_test = np.load('X_test.npy', mmap_mode='r+')
y = np.load('y_train.npy')



"""print('X_shape =', X_train.shape)
print('X_min =',X_train.min())
print('X_mean =',X_train.mean())
print('X_std =',X_train.std())
print('X_max =',X_train.max())"""
#Observe the characteristics of X_train data


"""print('y_shape =', y.shape)
print('y_min =',y.min())
print('y_mean =',y.mean())
print('y_std =',y.std())
print('y_max =',y.max())"""#Observe the characteristics of Y data

def describeData(a, b):
    print('Total number of images: {}'.format(len(a)))
    # Total number of data images
    print('Number of Type0 Images: {}'.format(np.sum(b == 0)))
    # The number of images whose type is 0
    print('Number of Type1 Images: {}'.format(np.sum(b == 1)))
    # The number of images whose type is 1
    print('Number of Type2 Images: {}'.format(np.sum(b == 2)))
    # The number of images whose type is 2
    print('Number of Type3 Images: {}'.format(np.sum(b == 3)))
    # The number of images whose type is 3
    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))
    # Data sample length, width, and number of channels

"""describeData(X_train, y)"""# Observe category distribution and base case

def plot1(x,y,c):# View an image of the selected category
    t0 = x[y==3]
    plt.title(3)
    plt.subplot(330 + c)
    plt.imshow(t0[c])


for i in range(1,10,1): #View an image of category 3
    plot1(X_train,y,i)

plt.show()

def plot2(a):# Draw the histogram
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(a)
    plt.axis('off')
    histo = plt.subplot(1, 2, 2)
    histo.set_ylabel('Count') # set y axis to total
    histo.set_xlabel('Frequency')  #Sets the x axis to the frequency of pixels
    if y[0]:
        plt.title('type0')
    elif y[1]:
        plt.title('type1')
    elif y[2]:
        plt.title('type2')
    elif y[3]:
        plt.title('type3')
    # Draw R channel
    plt.hist(a[:, :, 0].flatten(), bins=30, lw=0, color='r', alpha=0.5);
    # Draw G channel
    plt.hist(a[:, :, 1].flatten(), bins=30, lw=0, color='g', alpha=0.5);
    # Draw B channel
    plt.hist(a[:, :, 2].flatten(), bins=30, lw=0, color='b', alpha=0.5);
    plt.show()

"""plot2(X_train[1])"""