import math
from re import M
import psutil
from memory_profiler import profile

import os


import numpy as np
import time
from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft
from sklearn import metrics

import sys


# z-normalized
def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    # if mns.dim different with a.dim
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)


def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res

# Cross-correlation 2m  - 1
def _ncc_c(x, y):
    """
    >>> _ncc_c([1,2,3,4], [1,2,3,4])
    array([ 0.13333333,  0.36666667,  0.66666667,  1.        ,  0.66666667,
            0.36666667,  0.13333333])
    >>> _ncc_c([1,1,1], [1,1,1])
    array([ 0.33333333,  0.66666667,  1.        ,  0.66666667,  0.33333333])
    >>> _ncc_c([1,2,3], [-1,-1,-1])
    array([-0.15430335, -0.46291005, -0.9258201 , -0.77151675, -0.46291005])
    """
    #norm(x, ord = None, axis = None, keepdims = False) 
    den = np.array(norm(x) * norm(y))
    #print("den: ",den)
    den[den == 0] = np.Inf

    x_len = len(x)
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    return np.real(cc) / den


def _ncc_c_2dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional X arrays and 1 dimensional
    y vector

    Returns a 2 dimensional array of normalized fourier transforms
    """
    den = np.array(norm(x, axis=1) * norm(y))
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[:,-(x_len-1):], cc[:,:x_len]), axis=1)
    return np.real(cc) / den[:, np.newaxis]


def _ncc_c_3dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional X arrays and 2 dimensional
    y vector

    Returns a 3 dimensional array of normalized fourier transforms
    """
    den = norm(x, axis=1)[:, None] * norm(y, axis=1)
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    #kkk
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size))[:, None])
    cc = np.concatenate((cc[:,:,-(x_len-1):], cc[:,:,:x_len]), axis=2)
    return np.real(cc) / den.T[:, :, None]


def _sbd(x, y):
    """
    >>> _sbd([1,1,1], [1,1,1])
    (-2.2204460492503131e-16, array([1, 1, 1]))
    >>> _sbd([0,1,2], [1,2,3])
    (0.043817112532485103, array([1, 2, 3]))
    >>> _sbd([1,2,3], [0,1,2])
    (0.043817112532485103, array([0, 1, 2]))
    """
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    #kkk
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return dist, yshift


def _extract_shape(idx, x, j, cur_center):
    
    """
    >>> _extract_shape(np.array([0,1,2]), np.array([[1,2,3], [4,5,6]]), 1, np.array([0,3,4]))
    array([-1.,  0.,  1.])
    >>> _extract_shape(np.array([0,1,2]), np.array([[-1,2,3], [4,-5,6]]), 1, np.array([0,3,4]))
    array([-0.96836405,  1.02888681, -0.06052275])
    >>> _extract_shape(np.array([1,0,1,0]), np.array([[1,2,3,4], [0,1,2,3], [-1,1,-1,1], [1,2,2,3]]), 0, np.array([0,0,0,0]))
    array([-1.2089303 , -0.19618238,  0.19618238,  1.2089303 ])
    >>> _extract_shape(np.array([0,0,1,0]), np.array([[1,2,3,4],[0,1,2,3],[-1,1,-1,1],[1,2,2,3]]), 0, np.array([-1.2089303,-0.19618238,0.19618238,1.2089303]))
    array([-1.19623139, -0.26273649,  0.26273649,  1.19623139])
    """
    _a = []
    for i in range(len(idx)):
        if idx[i] == j:
            # init case
            if cur_center.sum() == 0:
                opt_x = x[i]
            else:
                _, opt_x = _sbd(cur_center, x[i])
            _a.append(opt_x)
    a = np.array(_a)

    if len(a) == 0:
        return np.zeros((1, x.shape[1]))
    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)
    s = np.dot(y.transpose(), y)

    p = np.empty((columns, columns))
    p.fill(1.0/columns)
    p = np.eye(columns) - p

    m = np.dot(np.dot(p, s), p)
    _, vec = eigh(m)
    centroid = vec[:, -1]
    finddistance1 = math.sqrt(((a[0] - centroid) ** 2).sum())
    finddistance2 = math.sqrt(((a[0] + centroid) ** 2).sum())

    if finddistance1 >= finddistance2:
        centroid *= -1

    return zscore(centroid, ddof=1)


def _kshape(x, k):
    """
    >>> from numpy.random import seed; seed(0)
    >>> _kshape(np.array([[1,2,3,4], [0,1,2,3], [-1,1,-1,1], [1,2,2,3]]), 2)
    (array([0, 0, 1, 0]), array([[-1.2244258 , -0.35015476,  0.52411628,  1.05046429],
           [-0.8660254 ,  0.8660254 , -0.8660254 ,  0.8660254 ]]))
    """

    '''
    start_time = time.time()
    x = zscore(x,axis=1)
    end_time = time.time()  # 记录程序结束运行时间
    print('zscore Took %f second' % (end_time - start_time))
    '''
    x = zscore(x,axis=1)
    m = x.shape[0]
    idx = randint(0, k, size=m)
    centroids = np.zeros((k, x.shape[1]))
    distances = np.empty((m, k))
    
    iternum = 50
    print("start iter")
    for i in range(iternum):
        
        print("iter i ------------>", i)
        old_idx = idx
        print("start extract_shape")
        start_time = time.time()
        for j in range(k):
            centroids[j] = _extract_shape(idx, x, j, centroids[j])
        end_time = time.time()  
        print('extract_shape Took %f second' % (end_time - start_time))

        print("start ncc_c_3dim")
        start_time = time.time()
        distances = (1 - _ncc_c_3dim(x, centroids).max(axis=2)).T
        end_time = time.time()  
        print('ncc_c_3dim Took %f second' % (end_time - start_time))

        print("start array_equal")
        idx = distances.argmin(1)
        if np.array_equal(old_idx, idx):
            iternum = i
            break
        
        print('\n')
    return idx, centroids, iternum

#@profile
def kshape(x, k):
    idx, centroids, iternum = _kshape(np.array(x), k)
    clusters = []
    for i, centroid in enumerate(centroids):
        series = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        clusters.append((centroid, series))
    return idx,clusters,iternum


def fileReader(filename):
    my_list = []
    with open(filename, 'r') as infile:
        data = infile.readlines()  
 
        for line in data:
            odom = line.split(",")        
            my_list.append(odom)
    return my_list
    
def fileReader2(filename):
    my_list = []
    with open(filename, 'r') as infile:
        data = infile.readlines()  
 
        for line in data:
            #odom = line.split(",")
            odom = line.split("	")         
            my_list.append(odom)
    return my_list



#@count_info

def labeled_process(filename):
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
    #k = para
    timeAbout = []
    timeAbout.append(filename)
    start_time = time.time()
    idx,centers,iternum = kshape(data, k)
    end_time = time.time() 
    timeCost = end_time - start_time
    print('total time Took %f second' % (timeCost))
    timeAbout.append(timeCost/iternum)
    kvalue = str(k)
    timeAbout.append(kvalue)
    
    f=open("runtimesingleCPU.txt","a")
    f.writelines(str(timeAbout))
    f.close()
    #print("centers",centers)
    Path=os.path.dirname(os.getcwd())
    np.savetxt(Path+"/out/original.txt", idx, delimiter=",", fmt='%.0f')

    #print("centers",centers)

    '''
    labels_true = labe.reshape(labe.shape[0]*labe.shape[1],)
    labels_true = labels_true.astype(int)
    labels_true = labels_true.tolist()
    labels_pred = idx.tolist()
    rand_score = metrics.rand_score(labels_true, labels_pred)
    print("rand_score",rand_score)
    '''

#@count_info

def unlabeled_process(filename, k):
    m = fileReader(filename)

    for i in range(len(m)):
        del m[i][len(m[i])-1]
        for j in range(len(m[0])):
            m[i][j] = float(m[i][j])
    
    mat = np.array(m)

    timeAbout = []
    timeAbout.append(filename)
    start_time = time.time()
    idx,centers,iternum = kshape(mat, k)
    end_time = time.time()  # 记录程序结束运行时间
    timeCost = end_time - start_time
    print('total time Took %f second' % (timeCost))
    timeAbout.append(timeCost/iternum)
    kvalue = str(k)
    timeAbout.append(kvalue)
    
    f=open("runtimeCPU.txt","a")
    f.writelines(str(timeAbout))
    f.close()
    #print("centers",centers)
    Path=os.path.dirname(os.getcwd())
    np.savetxt(Path+"/out/original.txt", idx, delimiter=",", fmt='%.0f')

    
    '''
    #labelname =  Path +"/data/MosquitoSound/MosquitoSound_TRAIN_Lable"
    #labelname =  Path +"/data/InsectSound/InsectSound_TRAIN_Lable"
    #labelname =  Path +"/data/WhaleCalls/RightWhaleCalls_TRAIN_Lable"
    labelname =  Path +"/data/CatsDogs/CatsDogs_TRAIN_Lable"
    #labelname =  Path +"/data/FruitFlies/FruitFlies_TRAIN_Lable"
    m = fileReader(labelname)
    labels_true = np.array(m)
    labels_pred = idx
    labels_true = labels_true.reshape(labels_true.shape[0]*labels_true.shape[1],)
    labels_true = labels_true.astype(int)
    labels_true = labels_true.tolist()
    #print(labels_true)
    #print(labels_pred)
    print(len(labels_true))
    print(len(labels_pred))
    
    rand_score = metrics.rand_score(labels_true, labels_pred)
    print("rand_score",rand_score)
    '''



#python testkshape.py filePath k label_or_unlabel_flag
#python testkshape.py /home/songruibao/code/data/InsectSound/InsectSound_TRAIN 10 1

if __name__ == "__main__":
    filePath = sys.argv[1]
    k =  int(sys.argv[2])
    flag =  int(sys.argv[3])
    #unlabeled_process(filename27, 2)
    if(flag == 1):
        unlabeled_process(filePath, k)
    if(flag == 0):
        labeled_process(filePath)
    
    
