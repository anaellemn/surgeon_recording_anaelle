import sys
from os.path import join
import pandas as pd
import numpy as np
import cv2
from threading import Thread, Event
from multiprocessing import Lock
import os
import asyncio
import time
from surgeon_recording.sensor_handlers.camera_handler import CameraHandler


class Reader(object):
    def __init__(self):
        self.sensor_list = ["camera", "emg", "optitrack", "tps"]
        self.data = {}
        self.images = {}
        for s in self.sensor_list:
            self.data[s] = []
        self.mutex = Lock()
        self.blank_image = CameraHandler.create_blank_image(encode=True)
        self.data_changed = False
        self.stop_event = Event()
        self.image_extractor_thread = Thread(target=self.extract_images)
        self.image_extractor_thread.daemon = True
        self.image_extractor_thread.start()

    def get_experiment_list(self, data_folder):
        res = {}
        needed_files = [s + '.csv' for s in self.sensor_list] + ['rgb.avi', 'depth.avi']
        exp_list = []
        exp_list = [x[0] for x in os.walk(data_folder) if all(item in x[2] for item in needed_files)]
        for exp in exp_list:
            res[exp] = exp
        return res

    def get_indexes(self, initial_guesses, idx):
        max_index = self.get_nb_frames()
        camera_index = initial_guesses["camera"][idx]
        camera_frame = self.data["camera"].iloc[camera_index]
        time = camera_frame[1]
        # extract the other sensor data (exept camera)
        indexes = {}
        indexes["camera"] = camera_index
        for s in self.sensor_list[1:]:
            index = initial_guesses[s][idx]
            frame = self.data[s].iloc[index]
            indexes[s] = index
            prev_sign = np.sign(time - frame[1])
            while abs(time - frame[1]) > 1e-3:
                sign = np.sign(time - frame[1])
                if prev_sign - sign != 0:
                    break
                index = int(index + sign * 1)
                if index < 0 or index >= self.get_nb_sensor_frames(s):
                    break
                frame = self.data[s].iloc[index]
                indexes[s] = index
                prev_sign = sign
        return indexes

    def get_window_data(self, indexes):
        start_frame = indexes["camera"][0]
        stop_frame = indexes["camera"][1]
        max_index = self.get_nb_frames()
        if start_frame < 0 or start_frame > max_index:
            print("Incorrect index, expected number between 0 and " + str(max_index) + " got " + str(start_frame))
            return -1
        if stop_frame < 0 or stop_frame > max_index:
            print("Incorrect index, expected number between 0 and " + str(max_index) + " got " + str(stop_frame))
            return -1
        start_indexes = self.get_indexes(indexes, 0)
        stop_indexes = self.get_indexes(indexes, 1)
        window_data = {}
        for s in self.sensor_list:
            window_data[s] = self.data[s].iloc[start_indexes[s]:stop_indexes[s]+1]
        return window_data

    def get_nb_frames(self):
        return len(self.data["camera"])

    def get_nb_sensor_frames(self, sensor):
        return self.data[sensor].count()[0]

    def init_image_list(self):
        for t in ["rgb", "depth"]:
            self.images[t] = []
            for i in range(len(self.data["camera"])):
                self.images[t].append(self.blank_image)

    def extract_images(self):
        def extract_image(video):
            _, frame = video.read()
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer

        while True:
            if self.data_changed:
                self.data_changed = False
                rgb_video = cv2.VideoCapture(join(self.exp_folder, "rgb.avi"))
                depth_video = cv2.VideoCapture(join(self.exp_folder, "depth.avi"))
                for i in range(self.get_nb_frames()):
                    rgb_image = extract_image(rgb_video)
                    depth_image = extract_image(depth_video)
                    with self.mutex:
                        if self.data_changed:
                            break
                        self.images["rgb"][i] = rgb_image
                        self.images["depth"][i] = depth_image
                rgb_video.release()
                depth_video.release()
            time.sleep(0.01)

    def get_image(self, video_type, frame_index):
        max_index = self.get_nb_frames()
        if frame_index < 0 or frame_index > max_index:
            print("Incorrect index, expected number between 0 and " + str(max_index) + " got " + str(frame_index))
            return -1
        with self.mutex:
            image = self.images[video_type][frame_index]
        return image

    def align_relative_time(self):
        min_time = max([self.data[s]['relative_time'].iloc[0] for s in self.sensor_list])
        max_time = min([self.data[s]['relative_time'].iloc[-1] for s in self.sensor_list])

        for s in self.sensor_list:
            self.data[s] = self.data[s][np.logical_and(self.data[s]['relative_time'] - min_time > 1e-3, self.data[s]['relative_time'] - max_time < 1e-3)]
            self.data[s]['relative_time'] -= min_time

        start_camera_index = self.data['camera'].index[0]-1
        stop_camera_index = self.data['camera'].index[-1]
        self.offset = start_camera_index
        for image in ['rgb', 'depth']:
            self.images[image] = self.images[image][start_camera_index:stop_camera_index]

    def play(self, exp_folder):
        self.exp_folder = exp_folder
        indexes = {}
        for s in self.sensor_list:
            self.data[s] = pd.read_csv(join(exp_folder, s + ".csv"))
        with self.mutex:
            self.init_image_list()
            self.align_relative_time()
            self.data_changed = True
        
    def export(self, folder, indexes):
        export_folder = join(self.exp_folder, folder)
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        start_index = indexes["camera"][0]
        stop_index = indexes["camera"][1]
        cut_camera_data = self.data["camera"].iloc[start_index:stop_index+1]
        cut_camera_data.to_csv(join(export_folder, 'camera.csv'))
        print("Camera data file exported")
        cut_data = self.get_window_data(indexes)
        for key, value in cut_data.items():
            value.to_csv(join(export_folder, key + '.csv'))
            print(key + " data file exported")
        self.export_video(export_folder, start_index, stop_index)
        print("Export complete")

    def export_video(self, folder, start_index, stop_index):
        for t in ["rgb", "depth"]:
            original_video = cv2.VideoCapture(join(self.exp_folder, t + '.avi'))
            cut_video = cv2.VideoWriter(join(folder, t + '.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)
            for i in range(self.offset + stop_index + 1):
                _, frame = original_video.read()
                if i < self.offset + start_index:
                    continue
                cut_video.write(frame)
            print(t + " video file exported")
            original_video.release()
            cut_video.release()