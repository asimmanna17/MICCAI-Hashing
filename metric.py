import math
import numpy as np

def mAP(q_name, sorted_pool):
    ret_classes = [sorted_pool[i][0].split("_")[0:3] for i in range(len(sorted_pool))]
    q_class = q_name.split(".")[0].split("_")[0:3]
    #print(ret_classes)
    #print(q_class)
    initlist = [int(q_class == i) for i in ret_classes]
    #print(initlist)
    den = np.sum(initlist)
    #print(den)
    if den == 0:
        return 0
    x = 0
    preclist = [0]*len(initlist)
    for idx, pts in enumerate(initlist):
        x += pts #rel(n)
        preclist[idx] = x/(idx+1) #rel(n)/k
    #print(preclist)
    num = np.dot(preclist, initlist)
    #print(num)
    #print(num/den)
    return num/den