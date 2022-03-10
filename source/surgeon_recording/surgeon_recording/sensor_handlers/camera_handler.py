import pyrealsense2 as rs
import numpy as np
import zmq
import cv2
import csv
import os
from os.path import join
import time
from surgeon_recording.sensor_handlers.sensor_handler import SensorHandler

class CameraHandler(SensorHandler):  #hérite de la class sensorhandler
    def __init__(self, parameters):
        SensorHandler.__init__(self, 'camera', parameters)                                #initialise les param de la camera avec fichier sensor handler

        if self.running:                                                                  #si statut on ou simulated on initialise
            self.color_image = []
            self.depth_colormap = []
            if not self.simulated:                                                        #si not simulated (donc on)
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                try:
                    self.pipeline.start(self.config)
                except:
                    print("Error initializing the camera")
            self.colorwriter = None
            self.depthwriter = None

    @staticmethod
    def get_parameters():
        parameters = SensorHandler.read_config_file('camera')    #get param from configuration file
        if parameters['status'] != 'off':                        #si la camera n'est pas off alors on update le Headers dans les param
            parameters.update({ 'header': [] })
        return parameters

    @staticmethod
    def create_blank_image(encode=False):
        image = np.zeros((480, 640, 3), np.uint8)               #on definit nombre de pixels et les 3 plaes pour RGB dans chaque
        # Since OpenCV uses BGR, convert the color first
        color = tuple((0, 0, 0))                                #liste non modifiable
        # Fill image with color
        image[:] = color                                        #on met R=0 G=0 et B=0 pour chaque pixel = image blanche            
        if not encode:                                          #si on ne veut pas encoder (donc encode est false)
            return image
        return cv2.imencode('.jpg', image)[1]                   #sinon on encode

    def acquire_data(self):
        if not self.simulated:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()                          #ou sont ces fonctions ?
            if not depth_frame or not color_frame:
                return

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            self.color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        else:
            self.color_image = CameraHandler.create_blank_image()           #simulé: on crée blank image
            self.depth_colormap = CameraHandler.create_blank_image()
            time.sleep(0.03)                                                #pause

        absolute_time = time.time()                                         #prend le time actuel
        data = [self.index + 1, absolute_time, absolute_time - self.start_time]   #met index, absolute time et le relative time aussi 
        self.index = data[0]
        return data

    def send_data(self, topic, data):
        super().send_data(topic, data)                        #appelle de la classe parente = sensor Handler
        super().send_data('rgb', self.color_image)
        super().send_data('depth', self.depth_colormap)

    @staticmethod
    def display_images(color_image, depth_image):
        # Stack both images horizontally
        images = np.hstack((color_image, depth_image))
        # # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)    #crée une fenetre qui s'appelle RealSense, cv2.window_autosize permet d'avoir la size automatic
        cv2.imshow('RealSense', images)                      #montre image
        cv2.waitKey(1)

    def setup_recording(self, recording_folder, start_time):
        with self.lock:
            if not os.path.exists(recording_folder):
                os.makedirs(recording_folder)
            color_path = join(recording_folder, 'rgb.avi')
            depth_path = join(recording_folder, 'depth.avi')
            self.colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)  #write video at color_path, with code to compress frame specified (fourcc)
            self.depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)
        super().setup_recording(recording_folder, start_time)

    def stop_recording(self):
        super().stop_recording()
        with self.lock:
            if self.colorwriter is not None:
                self.colorwriter.release()          #release les lock
            if self.depthwriter is not None:
                self.depthwriter.release()

    def record(self, data):
        super().record(data)
        if self.recording:
                                                # write the images
            if self.colorwriter is not None:
                self.colorwriter.write(self.color_image)
            if self.depthwriter is not None:
                self.depthwriter.write(self.depth_colormap)

    def shutdown(self):
        super().shutdown()
        if not self.simulated:
            self.pipeline.stop()
        print("camera closed cleanly")


def main(args=None):
    parameters = CameraHandler.get_parameters()
    camera_handler = CameraHandler(parameters)
    camera_handler.run()
    
if __name__ == '__main__':
    main()