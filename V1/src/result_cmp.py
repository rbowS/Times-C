from sklearn import metrics
import numpy as np
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

def fileReader2(filename):
    my_list = []
    with open(filename, 'r') as infile:
        data = infile.readlines()  
 
        for line in data:    
            odom = line.split("	")         
            my_list.append(odom)
    return my_list


#python result_cmp.py labelPath resultPath label_or_unlabel_flag
#python result_cmp.py /home/gpu2/Desktop/srbWorkSpace/tensorCore/proj_1/data/InsectSound/InsectSound_TRAIN_Lable /home/gpu2/Desktop/srbWorkSpace/tensorCore/V1/out/cukshape/result.txt 1
if __name__ == "__main__":
    labelname = sys.argv[1]
    outname =  sys.argv[2]
    flag =  int(sys.argv[3])
    savename = "rand_index"
    m = fileReader(labelname)
    out = fileReader(outname)
    labels_pred = []

    for i in range(len(out[0])):
        if out[0][i] != '':
            labels_pred.append(int(float(out[0][i])))

    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] = float(m[i][j])
    
    m = np.array(m)
    
    if(flag == 0):
        labels_true = m[:,:1]
    
    if(flag == 1):
        labels_true = m
    
    
    labels_true = labels_true.reshape(labels_true.shape[0]*labels_true.shape[1],)
    labels_true = labels_true.astype(int)
    labels_true = labels_true.tolist()
    #print(labels_true)
    #print(labels_pred)
    print(len(labels_true))
    print(len(labels_pred))
    
    rand_score = metrics.rand_score(labels_true, labels_pred)
    #rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    #mutual_score = metrics.mutual_info_score(labels_true, labels_pred)
    score = str(rand_score)
    print("rand_score",rand_score)
    #print("mutual_score",mutual_score)
    
    