import Augmentor
import os

def get_distortion_pipline(path, num):
    p = Augmentor.Pipeline(path)
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.sample(num)
    return p

if __name__ == "__main__":
    times = 2
    path = r"data\augtrain"
    num = len(os.listdir(path)) * times
    p = get_distortion_pipline(path, num)
