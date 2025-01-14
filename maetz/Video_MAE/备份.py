import time
from feature_extract import *
from dataset_creater import *
from dataset.write_data_label_txt_new import *
import os
import shutil
from pathlib import Path

# 遍历源文件夹和目标文件夹
def copy(source_folder, destination_folder):
    # 确保目标文件夹存在
    destination_folder = Path(destination_folder)
    if not destination_folder.exists():
        destination_folder.mkdir(parents=True, exist_ok=True)

    # 检查并列出源文件夹中的文件
    try:
        file_list = list(Path(source_folder).iterdir())
    except OSError as e:
        print(f"Error reading directory {source_folder}: {e}")
        return

    # 过滤出以 .npy 结尾的文件
    npy_files = [f for f in file_list if f.is_file() and f.suffix == '.npy']
    total_files = len(npy_files)

    # 移动文件
    with tqdm(total=total_files, desc="Moving files") as pbar:
        for file in npy_files:
            source_file = file
            destination_file = destination_folder / file.name
            shutil.move(source_file, destination_file)
            pbar.update(1)
def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def copy_files(source_folder, target_folder):
    file_list = os.listdir(source_folder)
    jpg_files = [filename for filename in file_list if filename.endswith('.jpg')]
    total_files = len(jpg_files)
    with tqdm(total=total_files, desc="Copying files") as pbar:
        for filename in jpg_files:
            source_file = os.path.join(source_folder, filename)
            target_file = os.path.join(target_folder, filename)
            shutil.copy2(source_file, target_file)
            pbar.update(1)

# 使用示例
if __name__ == '__main__':
    # 打开文件
    with open("D:\\VAD\\Video_MAE\\UBnormal_patch.txt", "r") as file:
        total_videos = sum(1 for line in file)  # 统计文件总行数，即总视频数量
        file.seek(0)  # 将文件指针重置到文件开头
        video_count = 0
        # 逐行读取文件内容
        for line in file:
            video_count += 1
            # 去除行末的换行符
            line = line.strip()
            # 使用下划线分割行内容
            elements = line.split('_')
            # 如果行内容不为空
            if elements:
                # 取出前三个元素作为scene，其余部分作为video
                scene = elements[1] + elements[2]
                video = line
                source_folder = f"D:\\VAD\\UBnormal_patch\\{scene}\\{video}"
                target_folder = r"D:\\VAD\\Video_MAE\\shanghaitech\\denseflow\\shanghaitech"
                delete_folder = "D:\VAD\\Video_MAE\\dataset\\shanghaitech\\features"
                delete_files_in_folder(target_folder)
                start_time = time.time()
                copy_files(source_folder, target_folder)
                end_time = time.time()

                execution_time = end_time - start_time
                print("Execution time:", execution_time)
                write_label()
                # 在 creat_i3d() 操作之前添加删除目标文件夹及其内部文件的操作
                if os.path.exists(delete_folder):
                    shutil.rmtree(delete_folder)
                i3d_function()
                creat_i3d()
                video = str(line)
                source_folders = 'D:\\VAD\\Video_MAE\\dataset\\shanghaitech\\features_video\\i3d\\combine\\shanghaitech'
                destination_folders = f'D:\\VAD\\MAE-UBnormal-Patch\\features_video\\i3d\\combine\\{video}'
                copy(source_folders, destination_folders)

                print(f'视频:{video}\n提取完毕')
            # 计算剩余视频数量
            remaining_videos = total_videos - video_count
            # 输出剩余视频数量
            print(f"剩余视频数量: {remaining_videos}\n-----------------------------------------------------")
