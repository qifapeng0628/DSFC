import os
import numpy as np
yuan_root = r'C:\Users\19557\Desktop\MAE_Net\UBnormal\features_video\i3d\combine'
all_name = os.listdir(yuan_root)
num=0
for name in all_name:
    k1=np.load(os.path.join(r'C:\Users\19557\Desktop\MAE_Net\UBnormal\features_video\i3d\combine',name,'feature.npy'))
    m1=np.load(os.path.join(r'D:\pengqf\UBnormal\features_video\i3d\combine',name,'feature.npy'))
    if k1.shape[0] != m1.shape[0]:
        num+=1
        print('name')
print('错误的个数:{}'.format(num))