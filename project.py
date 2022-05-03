import os
import torchvision
import numpy as np
import matplotlib.pyplot as plt

c10 = torchvision.datasets.CIFAR10(
    root='datasets',
    train=True,
    download=True
)

c10_test = torchvision.datasets.CIFAR10(
    root='datasets',
    train=False,
    download=True
)


def unpickle(file):
    import pickle as p
    with open(file, 'rb') as f:
        dictionary = p.load(f, encoding='latin1')
    return dictionary


data = unpickle('datasets/cifar-10-batches-py/test_batch')


def load_c_batch(file):
    import pickle as p
    with open(file, 'rb') as f:
        d = p.load(f, encoding='latin1')
        x = d['data']
        y = d['labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        y = np.array(y)
    return x, y


def load_c(root):
    xs = []
    ys = []
    for b in range(1, 2):
        f = os.path.join(root, 'data_batch_%d' % (b,))
        x, y = load_c_batch(f)
        xs.append(x)
        ys.append(y)
    xtr = np.concatenate(xs)
    ytr = np.concatenate(ys)
    del x, y
    xte, yte = load_c_batch(os.path.join(root, 'test_batch'))
    return xtr, ytr, xte, yte


plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

c_dir = 'datasets/cifar-10-batches-py'
x_train, y_train, x_test, y_test = load_c(c_dir)

# print('Training data shape: ', x_train.shape)
# print('Training labels shape: ', y_train.shape)
# print('Test data shape: ', x_test.shape)
# print('Test labels shape: ', y_test.shape)

filename = c_dir


def create_folder(filename):
    filename = filename.strip()
    filename = filename.rstrip("\\")
    isExists = os.path.exists(filename)

    if not isExists:
        os.makedirs(filename)
        print(filename+'Create sucessfully!')
        return  True
    else:
        print(filename+"Exist already!")
        return False


for i in range(1, 6):
    create_folder('datasets/cifar-10-batches-py/%d' % i)
    content = unpickle(filename + '/data_batch_' + str(i))
    print('load data...')
    print(content.keys())
    print('tranfering data_batch' + str(i))
    for j in range(10000):
        img = content['data'][j]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        img_name = 'datasets/cifar-10-batches-py/%d/%d.jpg' % (i, j)
        plt.imsave(img_name, img)
