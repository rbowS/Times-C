from scipy.io import arff
import pandas as pd
import os
import numpy as np

'''
name2class = {
    "RightWhale":1,
    "NoWhale":2
}
'''

name2class = {
    "Aedes_female":1,
    "Aedes_male":2,
    "Fruit_flies":3, 
    "House_flies":4, 
    "Quinx_female":5, 
    "Quinx_male":6,
    "Stigma_female":7,
    "Stigma_male":8,
    "Tarsalis_female":9, 
    "Tarsalis_male":10
}

'''
name2class = {
    "RightWhale":1,
    "NoWhale":2
}
'''



'''
name2class = {
    "melanogaster":1,
    "suzukii":2,
    "zaprionus":3
}
'''

'''
name2class = {
    "RightWhale":1,
    "NoWhale":2
}
'''

Path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(Path)

filepath = Path+"/data/InsectSound/InsectSound_TRAIN.arff"
filepath2 = Path+"/data/InsectSound/InsectSound_TEST.arff"


print("start load")
data = arff.loadarff(filepath)
data2 = arff.loadarff(filepath2)
print("start convert dataframe")
df = pd.DataFrame(data[0]).sample(frac=1)
df2 = pd.DataFrame(data2[0]).sample(frac=1)
df = pd.concat([df, df2])

sample = df.values[:, 0:len(df.values[0])-1]

# [b'1' b'-1' ...]
label = df.values[:, -1] 
cla = [] 
for i in label:
    #cla.append(name2class[i.decode()])
    cla.append(int(i))

np.savetxt(Path+"/data/phoneme/phoneme_DATA", sample, delimiter=",", fmt='%.6f')
np.savetxt(Path+"/data/phoneme/phoneme_Lable", cla, delimiter=",", fmt='%.0f')