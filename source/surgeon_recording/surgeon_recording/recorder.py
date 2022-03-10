import sys
import time
import numpy as np
import os
import cv2
from os.path import join
from threading import Thread
from threading import Event
import pandas as pd
from collections import deque
import zmq
from surgeon_recording.sensor_handlers.camera_handler import CameraHandler
from surgeon_recording.sensor_handlers.emg_handler import EMGHandler
from surgeon_recording.sensor_handlers.optitrack_handler import OptitrackHandler
from surgeon_recording.sensor_handlers.tps_handler import TPSHandler
from surgeon_recording.sensor_handlers.ft_sensor_handler import FTSensorHandler

class Recorder(object):
    def __init__(self, data_folder):
        self.recording = False
        self.data_folder = data_folder
        self.exp_folder = ''

        # init data storage
        self.data = {}
        self.buffered_images = {}
        self.init_data_buffer()

        # init all sensors
        self.sensor_list = ['camera', 'optitrack', 'emg', 'tps', 'ft_sensor']
        self.sensor_sockets = {}
        self.recorder_sockets = {}

        self.topics = {}
        self.topics['camera'] = ['rgb', 'depth']
        self.topics['optitrack'] = ['optitrack']
        self.topics['emg'] = ['emg']
        self.topics['tps'] = ['tps']
        self.topics['ft_sensor'] = ['ft_sensor']

        self.parameters = {}
        self.parameters['camera'] = CameraHandler.get_parameters()             #autre classe, importée au début, on utilise fonction de l'autre classe get param
        self.parameters['optitrack'] = OptitrackHandler.get_parameters()       #une classe par sensor
        self.parameters['emg'] = EMGHandler.get_parameters()
        self.parameters['tps'] = TPSHandler.get_parameters()
        self.parameters['ft_sensor'] = FTSensorHandler.get_parameters()

        for s in self.sensor_list:
            self.init_sensor(s)                                                   #initialise une case pour chaque sensor

        # initiliaze the threads                                                  #fil d'execution ?
        self.stop_event = Event()
        self.recording_threads = []
        self.recording_threads.append(Thread(target=self.get_camera_data))        #target =objet appelable ? on appelle les fonctions en même temps dans différents fil d'execution ?
        self.recording_threads.append(Thread(target=self.get_emg_data))
        self.recording_threads.append(Thread(target=self.get_optitrack_data))
        self.recording_threads.append(Thread(target=self.get_tps_data))
        self.recording_threads.append(Thread(target=self.get_ft_sensor_data))

        for t in self.recording_threads:
            t.start()                                                             #lance activité du fil d'execution

    def init_data_buffer(self):
        self.data['emg'] = deque(maxlen=2000)              #bounded to 2000 values: if add more than 2000, then values at the beginning are deleted to add the new values at the end
        self.data['optitrack'] = deque(maxlen=500)
        self.data['tps'] = deque(maxlen=500)
        self.data['ft_sensor'] = deque(maxlen=500)
        #INITIALISE LES BUFFER POUR CHAQUE SENSOR

    def init_recording_folder(self, folder):
        self.exp_folder = join(self.data_folder, folder)    #ajoute selected folder after data folder as exploration folder
        if not os.path.exists(self.exp_folder):
            os.makedirs(self.exp_folder)                    #créé si n'existe pas

    def init_sensor(self, sensor_name):                     #pour un sesor donné
        if self.parameters[sensor_name]['status'] == 'on' or self.parameters[sensor_name]['status'] == 'simulated' or self.parameters[sensor_name]['status'] == 'remote': #on regarde si le statut des param du sensor est ON ou SIMULATED ou REMOTE (donc pas off)
            ip = self.parameters[sensor_name]['streaming_ip'] if self.parameters[sensor_name]['streaming_ip'] != '*' else '127.0.0.1'   #on prend l'adresse ip, soit des param si existe, sinon adresse fixée
            port = self.parameters[sensor_name]['streaming_port']                                                                       #prend streaming port

            context = zmq.Context()                                               #create zmq Context
            self.sensor_sockets[sensor_name] = context.socket(zmq.SUB)            #create socket pour le sensor
            for t in self.topics[sensor_name]:                                    #nom du sensor ou rgb/depth pour la camera
                self.sensor_sockets[sensor_name].subscribe(str.encode(t))         #subscribe au nom encodé du sensor = ma socket reçoit les messages qui sont adressés a cette ref
            self.sensor_sockets[sensor_name].setsockopt(zmq.SNDHWM, 10)
            self.sensor_sockets[sensor_name].setsockopt(zmq.SNDBUF, 10*1024)      #socket options
            self.sensor_sockets[sensor_name].connect('tcp://%s:%s' % (ip, port))  #se connecte a l'adresse tcp://ip:port

            context = zmq.Context() 
            socket_recorder = context.socket(zmq.REQ)                             #nouvelle socket to record
            socket_recorder.connect('tcp://%s:%s' % (ip, port + 1))               #connecte adresse ip et port + 1 (port: chiffre)
            self.recorder_sockets[sensor_name] = socket_recorder                  #stock la socket pour le sensor actuel

    def get_camera_data(self):
        if 'camera' in self.sensor_sockets.keys():                                #si la socket pour la camera a été créée
            while not self.stop_event.is_set():                                   #tant qu'on veut pas arrêter
                data = CameraHandler.receive_data(self.sensor_sockets['camera'])  #autre classe pour recevoir data, via la socket
                self.buffered_images.update(data)                                 #ajoute les data qu'on reçoit par la socket au buffer de la camera

    def get_emg_data(self):
        if 'emg' in self.sensor_sockets.keys():
            while not self.stop_event.is_set():
                signal = EMGHandler.receive_data(self.sensor_sockets['emg'])
                for s in signal[self.topics['emg'][0]]:
                    self.data['emg'].append(s)                                    #pour les différents channel (?)

    def get_optitrack_data(self):
        if 'optitrack' in self.sensor_sockets.keys():
            while not self.stop_event.is_set():
                data = OptitrackHandler.receive_data(self.sensor_sockets['optitrack'])
                self.data['optitrack'].append(data[self.topics['optitrack'][0]])           #ajoute a data pour chaque sensor les data

    def get_tps_data(self):
        if 'tps' in self.sensor_sockets.keys():
            while not self.stop_event.is_set():
                data = TPSHandler.receive_data(self.sensor_sockets['tps'])
                self.data['tps'].append(data[self.topics['tps'][0]])

    def get_ft_sensor_data(self):
        if 'ft_sensor' in self.sensor_sockets.keys():
            while not self.stop_event.is_set():
                data = FTSensorHandler.receive_data(self.sensor_sockets['ft_sensor'])
                self.data['ft_sensor'].append(data[self.topics['ft_sensor'][0]])

    def get_buffered_data(self, sensor_name):
        header = ['index', 'absolute_time', 'relative_time'] + self.parameters[sensor_name]['header']
        data = self.data[sensor_name]
        if data:                                                             #si on a des data enregistrées (get_data) alors on les met dans une dataframe avec les index et les headers
            return pd.DataFrame(data=np.array(data)[:,1:], index=np.array(data)[:,0], columns=header[1:])
        return pd.DataFrame(columns=header[1:])                              #sinon on donne dataframe avec juste les headers (vide)

    def get_buffered_rgb(self):
        return cv2.imencode('.jpg', self.buffered_images['rgb'])[1]          #convert image format into streaming data = compression, current format is jpg

    def get_buffered_depth(self):
        return cv2.imencode('.jpg', self.buffered_images['depth'])[1]

    def record(self, folder):
        self.recording = True
        self.init_recording_folder(folder)
        self.init_data_buffer()
        message = {'recording': True, 'folder': self.exp_folder, 'start_time': time.time()}
        for key, s in self.recorder_sockets.items():
            s.send_json(message)                                             #send python object as a message using json to serialize = send message à la socket s
            s.recv_string()                                                  #close the socket (?)
    
    def stop_recording(self):
        self.recording = False
        message = {'recording': False}
        for key, s in self.recorder_sockets.items():
            s.send_json(message)
            s.recv_string()

    def shutdown(self):
        self.stop_event.set()
        for s in self.sensor_list:
            if s in self.sensor_sockets.keys():
                self.sensor_sockets[s].close()          #ferme les sockets
            if s in self.recorder_sockets.keys():
                self.recorder_sockets[s].close()
