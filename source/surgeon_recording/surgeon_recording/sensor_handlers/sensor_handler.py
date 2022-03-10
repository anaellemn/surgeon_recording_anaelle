import numpy as np
import zmq
import csv
from threading import Thread, Event, Lock
import os
from os.path import join
import time
import json
from abc import ABC, abstractmethod


class SensorHandler(ABC):
    def __init__(self, sensor_name, parameters):
        self.sensor_name = sensor_name
        self.running = (parameters['status'] == 'on' or parameters['status'] == 'simulated')   #true si on ou simulated
        self.simulated = parameters['status'] == 'simulated'                                   #true si simulated

        if self.running:
            self.header = parameters['header']
            ip = parameters['streaming_ip']
            port = parameters['streaming_port']

            self.recording = False
            self.index = 0
            self.start_time = time.time()                                         #définit le temps de départ au temps actuel
            self.timestep = 0.001 # security to not overfload the network
            
            # socket for publisher
            context = zmq.Context()
            self.socket = context.socket(zmq.PUB)
            self.socket.setsockopt(zmq.SNDHWM, 10)
            self.socket.setsockopt(zmq.SNDBUF, 10*1024)
            self.socket.bind('tcp://%s:%s' % (ip, port))
            # socker for recorder server
            self.recorder_socket = context.socket(zmq.REP)
            self.recorder_socket.bind('tcp://%s:%s' % (ip, port + 1))
            self.recorder_socket.setsockopt(zmq.LINGER, 0)

            self.stop_event = Event()
            self.recording_thread = Thread(target=self.recording_request_handler)
            self.recording_thread.start()
            self.lock = Lock()

    @staticmethod
    def read_config_file(sensor_name):
        filepath = os.path.abspath(os.path.dirname(__file__))
        with open(join(filepath, '..', '..', 'config', 'sensor_parameters.json'), 'r') as paramfile:
            config = json.load(paramfile)                                                                #on load le fichier des parametres
        if not sensor_name in config.keys():
            config[sensor_name] = {}
            config[sensor_name]['status'] = 'off'                                                       #si pas le sensor dans le fichier de config alors il est off
        return config[sensor_name]

    def generate_fake_data(self, dim, mean=0., var=1.):
        return np.random.normal(size=dim, loc=mean, scale=var)                                          #draw random samples from a normal distribution
    
    @abstractmethod
    def acquire_data(self):
        pass

    def send_data(self, topic, data):
        if len(data) > 0:                                         #si data n'est pas vide, on l'envoie avec les sockets (envoie topic en string et data en objet)
            self.socket.send_string(topic, zmq.SNDMORE)       
            self.socket.send_pyobj(data)

    @staticmethod
    def receive_data(socket):
        topic = socket.recv_string()            #quand on reçoit data: on met dans topic et data
        data = socket.recv_pyobj()
        return {topic: data}

    def recording_request_handler(self):
        while not self.stop_event.wait(0.01):
            if not self.recording:                #si false alors on attend commande
                print(self.sensor_name + ': waiting for commands')
            message = self.recorder_socket.recv_json()
            if message['recording'] and not self.recording:                  #si pas recording/simulated ET que le message reording est true
                self.setup_recording(message['folder'], message['start_time'])    #on prend le setup dans le message
                self.recorder_socket.send_string('recording started')             #envoie message que recording a commencé et on l'écrit
                print(self.sensor_name + ': recording started')
            elif not message['recording'] and self.recording:                #si recording est deja true et que faux dans message pour le recording, alors on arrete le recording (fonction et message et print)
                self.stop_recording()
                self.recorder_socket.send_string('recording stopped')
                print(self.sensor_name + ': recording stopped')
            else:
                self.recorder_socket.send_string('recording' if self.recording else 'not recording') #autre cas: on donne juste le statut, si recording ou pas recording (pas d'action de start ou stop)

    def setup_recording(self, recording_folder, start_time):
        with self.lock:
            if not os.path.exists(recording_folder):
                os.makedirs(recording_folder)
            f = open(join(recording_folder, self.sensor_name + '.csv'), 'w', newline='')              #open recording folder and csv sensor file
            self.writer = {'file': f, 'writer': csv.writer(f)}                             #return a writer object responsible for converting the user's data into delimited strings on the given file-like object
            self.writer['writer'].writerow(['index', 'absolute_time', 'relative_time'] + self.header)    #ensuite on peut écrire dans cet objet les noms des variables

            self.index = 0
            self.start_time = start_time
            self.recording = True                           #change variable a true pour qu'on commence le recording

    def stop_recording(self):
        with self.lock:
            self.recording = False                        #change variable a false
            self.index = 0
            self.start_time = time.time()
            self.writer['file'].close()                   #ferme le fichier

    def record(self, data):
        if self.recording:
            if data:
                if isinstance(data[0], list):               #si data est une liste
                    for d in data:
                        self.writer['writer'].writerow(d)  #on écrit une ligne avec chaque data
                else:
                    self.writer['writer'].writerow(data)   #sinon on écrit data tout d'un coup

    def shutdown(self):
        self.stop_event.set()     #on set la variable avec rien ?
        self.socket.close()
        self.recorder_socket.close()
        self.recording_thread.join()

    def run(self):
        if self.running:
            while True:
                try:
                    start = time.time()
                    with self.lock:
                        data = self.acquire_data()                        #fonctions acquire, record et send data
                        self.record(data)
                        self.send_data(self.sensor_name, data)
                    effective_time = time.time() - start             #tempas actuel moins temps départ boucle
                    wait_period = self.timestep - effective_time
                    if wait_period > 0:
                        time.sleep(wait_period)
                except KeyboardInterrupt:
                    print('Interruption, shutting down')
                    break
            self.shutdown()
