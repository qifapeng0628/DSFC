# coding:utf-8
import numpy as np
import os
import glob

def main(original_video_dict,dataset,dataset_mode):
    if dataset == 'UCF_Crime':
        zfill_number = 6
    else:
        zfill_number = 5 # ped1, ped2, shanghaitech or avenue
    if dataset_mode == 'i3d':
        if os.path.exists('./dataset/{}/{}'.format(dataset, dataset_mode)) == 0:
            os.makedirs('./dataset/{}/{}'.format(dataset, dataset_mode))
        with open(file='./dataset/{}/{}/rgb_list.txt'.format(dataset, dataset_mode), mode='w', encoding='utf-8') as f:
            with open(file='./dataset/{}/{}/label.txt'.format(dataset, dataset_mode), mode='w', encoding='utf-8') as t:
                for k, v in original_video_dict.items():
                    frames = os.listdir(os.path.join(v))
                    frames_number = int(len(frames))
                    framegt = np.zeros(shape=(frames_number), dtype='int8')
                    classgt = np.zeros(shape=(frames_number), dtype='int8')
                    for i in range(1, frames_number + 1):
                        f.write(os.path.join(v, 'img_' + str(i).zfill(zfill_number) + '.jpg' + '\n'))
                        t.write(str(framegt[i - 1]) + ':' + str(classgt[i - 1]) + '\n')
    else:
        if os.path.exists('./dataset/{}/{}'.format(dataset, dataset_mode)) == 0:
            os.makedirs('./dataset/{}/{}'.format(dataset, dataset_mode))
        with open(file='./dataset/{}/{}/rgb_list.txt'.format(dataset, dataset_mode),mode='w',encoding='utf-8') as f:
                with open(file='./dataset/{}/{}/label.txt'.format(dataset,dataset_mode),mode='w',encoding='utf-8') as t:
                    for k, v in original_video_dict.items():
                        frames = os.listdir(os.path.join(v))
                        frames_number = int(len(frames)/3)
                        framegt = np.zeros(shape=(frames_number), dtype='int8')
                        classgt = np.zeros(shape=(frames_number), dtype='int8')
                        for i in range(1, frames_number + 1, 1):
                            f.write(os.path.join(v, 'img_'+str(i).zfill(zfill_number)+ '.jpg'+'\n'))
                            t.write(str(framegt[i-1])+':'+str(classgt[i-1])+'\n')

def write_label():
    data_root = r'C:\Users\pengq\Desktop\maetz\Video_MAE'
    dataset = 'UCF_Crime'
    dataset_mode = 'i3d'  # i3d or c3d
    original_video = []
    original_video_dict = {}

    if dataset == 'LA':
        videopaths = glob.glob(os.path.join(data_root, dataset, 'denseflow', '*/*'))
    else:
        videopaths = glob.glob(os.path.join(data_root, dataset, 'denseflow', '*')) #ped1, ped2, shanghaitech or avenue
    videonames = []
    for videopath in videopaths:
        videoname = videopath.split('/')[-1]
        videonames.append(videoname)
        original_video_dict[videoname] = os.path.join(videopath)
    if os.path.exists('./dataset/{}/{}'.format(dataset, dataset_mode)) == 0:
        os.makedirs('./dataset/{}/{}'.format(dataset, dataset_mode))
    np.savetxt('./dataset/{}/{}/videoname.txt'.format(dataset, dataset_mode), np.asarray(videonames).reshape(-1),fmt='%s')
    main(original_video_dict=original_video_dict,
         dataset=dataset,
         dataset_mode=dataset_mode)
    # 定义文件路径
    file_path = 'C:\\Users\\pengq\\Desktop\\maetz\\Video_MAE\\dataset\\UCF_Crime\\i3d\\rgb_list.txt'
# 读取文件内容，去掉前两行
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
# 去掉前两行
    lines = lines[2:]
# 将剩余内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)


