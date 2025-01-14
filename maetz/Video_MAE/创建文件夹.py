import os

def create_subfolders_from_file(file_path, target_directory):
    # 读取文件中的文件名
    with open(file_path, 'r') as file:
        filenames = file.read().splitlines()

    # 在目标目录下创建子文件夹
    for filename in filenames:
        folder_path = os.path.join(target_directory, filename)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"创建文件夹: {folder_path}")
        else:
            print(f"文件夹已存在: {folder_path}")

# 指定文件路径和目标目录
file_path = r'C:\Users\pengq\Desktop\maetz\Video_MAE\ucfp3.txt'
target_directory = r'C:\Users\pengq\Desktop\maetz\UCF_Crime\features_video\i3d\combine'
create_subfolders_from_file(file_path, target_directory)
