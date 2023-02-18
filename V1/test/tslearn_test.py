# Author: Romain Tavenard
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from tslearn.clustering import KShape
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn import metrics
from tslearn.clustering import KernelKMeans
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
import time
import os
import sys

def fileReader(filename):
    my_list = []
    with open(filename, 'r') as infile:
        data = infile.readlines()  
 
        for line in data:
            odom = line.split(",")        
            my_list.append(odom)
    return my_list



def labeled_process(filename,savename):
    m = fileReader(filename)
    #m = fileReader2(filename)

    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] = float(m[i][j])
    
    mat = np.array(m)
    data = mat[:,1:]
    labe = mat[:,:1]

    k = int(np.max(labe))
    u = int(np.min(labe))
    if u == 0:
        k += 1
    
    methodinfo = []
    method = "KShape"
    #method = "softdtwKMeans"
    #method = "EDKMeans"
    str1 = method+"<-------->"+savename
    methodinfo.append(str1)
    print(str1)
    start_time = time.time()
    ks = KShape(n_clusters=k, max_iter=50, verbose=True)
    idx = ks.fit_predict(data)
    '''
    ks = KernelKMeans(n_clusters=k,
                      kernel="gak",
                      kernel_params={"sigma": "auto"},
                      n_init=20,
                      verbose=True)
    
    idx = ks.fit_predict(data)
    
    km = TimeSeriesKMeans(n_clusters=k, verbose=True)
    idx = km.fit_predict(data)
    '''
    
    # max_iter=2
    '''
    dba_km = TimeSeriesKMeans(n_clusters=k,
                          metric="softdtw",
                          verbose=True)
    idx = dba_km.fit_predict(data)
    '''
    
    '''
    sdtw_km = TimeSeriesKMeans(n_clusters=k,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True)
    idx = sdtw_km.fit_predict(data)
    '''
    
    end_time = time.time()  # 记录程序结束运行时间
    print('total time Took %f second' % (end_time - start_time))
    timecost = end_time - start_time
    print('total time Took %f second' % (timecost))
    timecostinfo = str(timecost)+"sec"
    methodinfo.append(timecostinfo)
    Path=os.path.dirname(os.getcwd())
    np.savetxt(Path+"/out/tslearn/"+method+"_"+savename+".txt", idx, delimiter=",", fmt='%.0f')
    #print("centers",centers)



def unlabeled_process(filename,k,savename):
    m = fileReader(filename)

    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] = float(m[i][j])
    
    data = np.array(m)
    methodinfo = []
    method = "KShape"
    str1 = method+"<-------->"+savename
    methodinfo.append(str1)
    print(str1)
    start_time = time.time()
    ks = KShape(n_clusters=k, max_iter=50, verbose=True)
    #ks = KernelKMeans(n_clusters=k)
    idx = ks.fit_predict(data)
    #km = TimeSeriesKMeans(n_clusters=k, verbose=True)
    #idx = km.fit_predict(data)
    '''
    km = TimeSeriesKMeans(n_clusters=k,
                          metric="dtw",
                          verbose=True)
    idx = km.fit_predict(data)
    '''
    #km = TimeSeriesKMeans(n_clusters=k, verbose=True)
    #idx = km.fit_predict(data)
    '''
    sdtw_km = TimeSeriesKMeans(n_clusters=k,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True)
    idx = sdtw_km.fit_predict(data)
    '''
    end_time = time.time()  # 记录程序结束运行时间
    timecost = end_time - start_time
    print('total time Took %f second' % (timecost))
    timecostinfo = str(timecost)+"sec"
    methodinfo.append(timecostinfo)
    #print("centers",centers)
    Path=os.path.dirname(os.getcwd())
    np.savetxt(Path+"/out/tslearn/"+method+"_"+savename+".txt", idx, delimiter=",", fmt='%.0f')

    


#python tslearn_test.py filePath k label_or_unlabel_flag
#python tslearn_test.py /home/songruibao/code/data/InsectSound/InsectSound_TRAIN 10 1

if __name__ == "__main__":
    filePath = sys.argv[1]
    k =  int(sys.argv[2])
    flag =  int(sys.argv[3])
    savename = "tslearn"
    #unlabeled_process(filename27, 2)
    if(flag == 1):
        unlabeled_process(filePath, k)
    if(flag == 0):
        labeled_process(filePath)


