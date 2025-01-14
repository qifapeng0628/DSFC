import cv2
import os
import numpy as np
import glob
import multiprocessing



def extract_frames(video_path):
    frames_dir = os.path.splitext(video_path)[0]  # get the video path without the extension
    os.makedirs(frames_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 1                                     # 将计数器从1开始
    while success:
        cv2.imwrite(os.path.join(frames_dir, "img_{:05d}.jpg".format(count)), image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
    return frames_dir

def compute_optical_flow(frames_dir):
    frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
    prev_frame = cv2.imread(frame_files[0])
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    for i, frame_file in enumerate(frame_files[1:]):
        curr_frame = cv2.imread(frame_file)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = compute_TVL1(prev_gray, curr_gray)
        flow_index = str(i+1).zfill(5)  # 根据索引生成光流图像的编号

        # 保存光流图像的x轴和y轴分量
        cv2.imwrite(os.path.join(frames_dir, "flow_x_{}.jpg".format(flow_index)), flow[:, :, 0])
        cv2.imwrite(os.path.join(frames_dir, "flow_y_{}.jpg".format(flow_index)), flow[:, :, 1])



        prev_gray = curr_gray





def compute_TVL1(prev, curr, bound=15):
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)

    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow

def process_video(video_path):
    frames_dir = extract_frames(video_path)
    compute_optical_flow(frames_dir)

if __name__ == '__main__':
    num_processes = 12  # Number of processes to run concurrently
    batch_size = 10  # Number of videos to process in each batch

    directory = f'F:\\UBnormal\\normal_scene_29\\normal_scene_29_scenario_3'  # 此处为Ubnormal-patch数据集的位置只需要改
    video_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.mp4')]

    for batch_start in range(0, len(video_files), batch_size):
        batch_end = min(batch_start + batch_size, len(video_files))
        batch_paths = video_files[batch_start:batch_end]

        # Process videos in the current batch concurrently
        pool = multiprocessing.Pool(processes=num_processes)
        pool.map(process_video, batch_paths)
        pool.close()
        pool.join()


