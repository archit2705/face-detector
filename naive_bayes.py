
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import glob
global idx2name, name2idx

idx2name = {}
name2idx = {}
on=1
zer=0

def gauss(x, mu, sigma):
    if sigma>0:
        return np.sqrt(1.0/2*np.pi*sigma*sigma)*np.exp( -1.0*(((x - mu)/sigma)**2) )
    return 0

def predict(point, means, std_dev, prior, classes):
    featureprob=[]
    feature_prob = np.zeros_like(means)
    for y in classes:
        for i in range(feature_prob.shape[0]):
            a=point[i]
            b=means[i, y]
            c=std_dev[i, y]
            feature_prob[i, y] = gauss(a,b,c)
    likelihood = np.zeros((feature_prob.shape[1], 1))
    lk=[]
    for y in classes:
        likelihood[y] = np.prod(feature_prob[np.nonzero(feature_prob), y])
    prediction=[]
    prediction = np.argmax([ np.asscalar(x*y) for x,y in zip(prior, likelihood) ])
    lk=prediction
    return lk

def read():
    trainfile=[]
    trainfile = sys.argv[1]
    testfile=[]
    testfile = sys.argv[2]
    nl=256*256
    train_images = np.full([on,nl],zer)
    train_labels = []
    num = zer
    with open(trainfile,'r') as files:
        lines = files.readlines()
        img_labs = []
        for line in lines:
            a = line.split(' ')
            img = np.asarray(Image.open(a[zer]).convert('L'))
            img = img.reshape((on,nl))
            train_images = np.concatenate((train_images, img), axis=zer)
            if a[on][:-on] not in name2idx:
                name2idx[a[on][:-on]] = num
                idx2name[num] = a[on][:-on]
                num=num+1
            train_labels.append(name2idx[a[on][:-on]])
        train_images = train_images[on:,:]
        train_img = []
        train_labels = np.array([train_labels]).T


    with open(testfile,'r') as f:
        lines = f.readlines()
        test_images = np.full([on,nl],zer)
        for line in lines:
            img = Image.open(line[:-on])
            b=np.zeros([1,256])  
            img = np.asarray(img.convert('L'))
            img = img.reshape((on, nl))
            test_images = np.concatenate((test_images, img), axis=zer)
        test_images = test_images[on:,:]

    return train_images, train_labels, test_images

def PCA(img, num_components):
    mu=img.mean(axis=zer)
    X = img - mu
    [r, c] = img.shape
    if r>=c:
        eig_val, eig_vec = np.linalg.eigh(X.T.dot(X))
    else:
        eig_val, eig_vec = np.linalg.eigh(X.dot(X.T))
        eig_vec = X.T.dot(eig_vec)

    for i in range(zer,eig_vec.shape[1],on):
        eig_vec[:,i] = eig_vec[:,i]/np.linalg.norm(eig_vec[:,i])
    idx = np.argsort(eig_val)[::-1]
    red_vec=np.zeros([1,num_components])
    eig_vec = eig_vec[:,idx]
    eig_val = eig_val[idx]
    reduced_vec = []
    eig_vec = eig_vec[:,:num_components]
    eig_val = eig_val[:num_components]

    reduced_vec = np.zeros([on, num_components])

    for i, l in enumerate(X):
        pr = l.dot(eig_vec)
        if pr.shape[zer] < num_components:
            all1=[]
            pr = np.array([np.pad(pr, (zer, num_components - pr.shape[zer]), 'constant', constant_values=zer)])
        b=[]
        reduced_vec = np.concatenate((reduced_vec, pr))
        b.append(reduced_vec)
    reduced_vec = reduced_vec[on:,:]
    vec2=[]
    return reduced_vec


def main():
    [train_images, label, test_images] = read()
    train = []
    train = PCA(train_images, 32)
    test=[]
    test = PCA(test_images, 32)
    classes=[]
    counts=[]
    classes, counts = np.unique(label, return_counts=True)
    num_feats = train.shape[1]
    num_classes = len(classes)

    prior = np.zeros((num_classes,1))
    for i,l in enumerate(counts):
        prior1=[]
        prior[classes[i]] = l * 1.0/train.shape[0]
    means=[]
    means = np.zeros((num_feats, num_classes))
    std_dev = np.zeros_like(means)
    variance=[]

    for y in classes:
        pt=0
        pts = train[np.where(label == y)[0]]
        for i in range(pts.shape[1]):
            mean1=[]
            means[i, y] = np.mean(pts[:, i])
            stdd=[] 
            std_dev[i, y] = np.std(pts[:, i])
    test1s=test
    for img in test1s:
	        print(idx2name[predict(img, means, std_dev, prior, classes)])

if __name__ == "__main__":
    main()