import numpy as np
import os
import glob
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import gzip
import pickle
import h5py
def combine_feature():
    import os
    import numpy as np

    # 文件夹路径
    folder_path = r'C:\Users\pengq\Desktop\maetz\Video_MAE\dataset\UCF_Crime\features\i3d\rgb\UCF_Crime'
    if os.path.exists(folder_path) == 0:
        os.makedirs(folder_path)
    # 读取文件并添加时间维度
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path)  # 读取.npy文件
            data = np.expand_dims(data, axis=0)  # 添加时间维度
            data_list.append(data)

    # 堆叠为一个数组
    stacked_data = np.concatenate(data_list, axis=0)

    # 输出堆叠后的数组形状
    print("堆叠后的数组形状:", stacked_data.shape)
    # 保存为pickle.gz文件并启用gzip压缩
    output_file = 'C:\\Users\\pengq\\Desktop\\maetz\\Video_MAE\\dataset\\UCF_Crime\\features_video\\i3d\\combine\\UCF_Crime\\feature.npy'
    np.save(output_file, stacked_data)
    print("已保存为.npy文件:", output_file)


def creat_i3d():
    combine_feature()



if __name__=='__main__':
    creat_i3d()

