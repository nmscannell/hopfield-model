#import hopfield
import scipy.io
digits = scipy.io.loadmat('digits.mat')
import numpy as np
import matplotlib.pyplot as plt

x = digits['train']
#x2 = digits['test'].astype('float32')/255
y = digits['trainlabels']
#y2 = digits['testlabels']
#for i in range(784):
#    for j in range(5000):
#        if x[i, j] < 0.5:
#            x[i, j] = -1
#        else:
#            x[i, j] = 1
#for i in range(784):
#    for j in range(1000):
#        if x2[i, j] < 0.5:
#            x2[i, j] = -1
#        else:
#            x2[i, j] = 1
zeros = []
ones = []
twos = []
threes = []
fours = []
fives = []
sixes = []
sevens = []
eights = []
nines = []
for i in range(5000):
    if y[i] == 0:
        zeros.append(i)
    elif y[i] == 1:
        ones.append(i)
    elif y[i] == 2:
        twos.append(i)
    elif y[i] == 3:
        threes.append(i)
    elif y[i] == 4:
        fours.append(i)
    elif y[i] == 5:
        fives.append(i)
    elif y[i] == 6:
        sixes.append(i)
    elif y[i] == 7:
        sevens.append(i)
    elif y[i] == 9:
        eights.append(i)
    elif y[i] == 9:
        nines.append(i)

ave_0 = x[:,zeros[0]]
for i in range(1, len(zeros)):
    ave_0 += x[:,zeros[i]]
print(ave_0)
ave_0 = np.true_divide(ave_0, len(zeros))
for i in range(len(ave_0)):
    if ave_0[i] < 0.5:
        ave_0[i] = -1
    else:
        ave_0[i] = 1
ave_1 = x[:,ones[0]]
for i in range(1, len(ones)):
    ave_1 += x[:,ones[i]]
print(ave_1)
ave_1 = np.true_divide(ave_1, len(ones))
for i in range(len(ave_1)):
    if ave_1[i] < 0.5:
        ave_1[i] = -1
    else:
        ave_1[i] = 1
print(ave_1)
print(ave_0.shape)
#print(x.shape)
im = np.copy(ave_0)
im = np.reshape(im, (28,28))
o = np.reshape(x[:,0], (28,28))
plt.imshow(im)
plt.show()
im = np.copy(ave_1)
im = np.reshape(im, (28,28))
plt.imshow(im)
plt.show()