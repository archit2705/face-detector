global idx2name, name2idx, W, X, y, reg, num_classes, num_train, zer

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import glob

zer=0
on=1
idx2name = {}
name2idx = {}
k=1000

def read():
    trainfile = sys.argv[1]
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
            if a[1][:-1] not in name2idx:
                name2idx[a[1][:-1]] = num
                idx2name[num] = a[1][:-1]
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
    return reduced_vec

def softmax(W, X, y, reg, num_classes, num_train):
    loss = 0.00
    a=W.shape[zer]
    b=W.shape[on]
    dW = np.zeros((a,b))

    for i in range(num_train):
        scores = X[i].dot(W)
        scores =scores-scores.max()
        scores_expsum = np.sum(np.exp(scores))
        cor_ex = np.exp(scores[y[i]])
        loss =loss - np.log( cor_ex / scores_expsum)
        dW[:, y[i]] =dW[:, y[i]]+np.array([(-on) * (scores_expsum - cor_ex) / scores_expsum * X[i]]).T
        for j in range(num_classes):
            if j == y[i]:
                pass
            dW[:, j] =dW[:, j]+ np.exp(scores[j]) / scores_expsum * X[i]

    loss =loss/num_train
    loss =loss+ reg * np.sum(W * W)
    dW =dW/ num_train
    dW =dW + 2 * reg * W
    return loss, dW

def descent(X, label, reg, num_classes, max_iter=k, ita=1e-5):
    W = np.zeros((X.shape[on], num_classes))
    losses = []
    while(max_iter):
        max_iter -= on
        loss, grad = softmax(W, X, label, reg, W.shape[on], X.shape[zer])
        losses.append(loss)
        W = W - ita*grad
    return W


def main():
    train_images, label, test_images = read()
    train = PCA(train_images, 32)
    test = PCA(test_images, 32)
    k=1000
    classes, _ = np.unique(label, return_counts=True)
    W = descent(train, label, k, len(classes))
    prediction = np.argmax(test.dot(W), axis=on)
    for lab in prediction:
        print(idx2name[lab])

if __name__ == "__main__":
    main()