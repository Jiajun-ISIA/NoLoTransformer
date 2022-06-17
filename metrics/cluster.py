
from sklearn.cluster import DBSCAN
from os import path
import numpy as np
import sys
import string
import time
import os
import shutil
import random
import pdb


#根据折线图确定聚类半径,计算欧式距离
def get_eps(x, pos):
    all_dist = 2 - 2 * np.dot(x, x.T)
    #print np.min(all_dist),np.max(all_dist)
    all_dist = np.clip(all_dist, 0, 4)
    all_dist = np.sqrt(all_dist)
    all_dist = np.sort(all_dist, axis=1)
    #print all_dist
    try:
        dist = all_dist[:, pos]
    except:
        dist = all_dist[:, -1]
    dist = np.sort(dist)

    sec_num = 10
    if dist.shape[0] < sec_num + 1:
        idx = int(dist.shape[0] * 0.7)
        return dist, idx

    slopes = []
    eps = []
    step = len(dist) / sec_num
    for i in range(sec_num):
        s = int(step * i)
        e = min(int(step * (i+1)), len(dist) - 1)
        if e == s:
            break
        slope = (dist[e] - dist[s]) / (e - s)
        slopes.append(slope)
        eps.append(dist[s])
    ind = np.argsort(slopes[3:])
    res_idx = ind[-1] + 3


    #logging.info("%s %s %s %s" % (slopes, res_idx, res_idx*step, eps[res_idx]))
    #print eps,res_idx
    return eps, res_idx


#基于超参进行聚类
def cluster(features,labels):
    keys = np.unique(labels)
    res = []
    for key in keys:
        feature = features[labels == key]
        epses, search_idx = get_eps(feature, pos=4)
        eps = epses[search_idx]
        #print eps
        num = feature.shape[0]
        min_samples_valid = 4
        if num >= min_samples_valid:
            min_samples = min_samples_valid
        else:
            min_samples = 1

        #min_samples = 4

        if eps <= 0:
            eps = 0.5
        print('clustering....')
        y_pred = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feature)
        #print y_pred
        
        cls_list = np.unique(y_pred)
    
        res += [len(cls_list)]
    pdb.set_trace()
    print(res)
    


# cnt = dict()
# for idx, lbl in enumerate(computed_cluster_labels): 
#     key = target_labels[idx][0]
#     if key not in cnt:
#         cnt[key] = [lbl]
#     elif lbl not in cnt[key]:
#         cnt[key] += [lbl]
#     for key in cnt: 
#         cnt[key] = len(cnt[key])
#     nums = []
#     for key in cnt:  
#         nums += [cnt[key]]

if __name__=='__main__':
    features = np.load('/home/lizhuo/Smooth_AP/src/feature.npy')
    labels = np.load('/home/lizhuo/Smooth_AP/src/target.npy')
    labels = labels.reshape(-1)
    cluster(features, labels)
