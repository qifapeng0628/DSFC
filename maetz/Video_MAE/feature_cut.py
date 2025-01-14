import os
import shutil
# 遍历源文件夹和目标文件夹
def copy(source_folders,destination_folders):
    for source_folder, destination_folder in zip(source_folders, destination_folders):
        # 获取源文件夹内的所有文件
        files = os.listdir(source_folder)

        # 遍历文件列表
        for file in files:
            # 构建源文件路径和目标文件路径
            source_file = os.path.join(source_folder, file)
            destination_file = os.path.join(destination_folder, file)

            # 将文件剪切到目标文件夹
            shutil.move(source_file, destination_file)


if __name__=='__main__':
    video = 'normal_scene_3_scenario_3_1'
    # 源文件夹路径，这个不用更改
    source_folders = [r'E:\AR\anomly_feature.pytorch-main\dataset\shanghaitech\features_video\i3d\combine\shanghaitech']

    print(f'本次生成的特征为: {video}')
    # 目标文件夹路径，这个是你准备存放抽取好的特征的文件夹位置
    destination_folders = [f'D:\\学习\\UBnormal-feature\\features_video\\i3d\\combine\\{video}']
    copy(source_folders, destination_folders)
