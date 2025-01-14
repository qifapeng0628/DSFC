import os

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.mp4'):
            file = os.path.join(directory, filename)
    return file

process_directory(r'F:\UBnormal\Scene1')





