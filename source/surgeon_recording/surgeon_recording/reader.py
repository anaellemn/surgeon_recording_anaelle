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
from shutil import copyfile
from surgeon_recording.sensor_handlers.camera_handler import CameraHandler
from os.path import exists

#quand on appelle une fonction de reader alors automatiquement c'est initialisé ?

class Reader(object):                                                                                      #create the object
    def __init__(self):                                                                                    #initiliaze the object     
        self.available_sensors = ['camera', 'emg', 'optitrack', 'tps', 'ft_sensor'] 
        self.needed_sensors = ['camera', 'emg']

        self.data = {}                                                                                     #self a une variable data qu'on pourra utiliser
        self.images = {}

        self.mutex = Lock()
        self.blank_image = CameraHandler.create_blank_image(encode=True)
        self.data_changed = False
        self.stop_event = Event()
        self.image_extractor_thread = Thread(target=self.extract_images)
        self.image_extractor_thread.daemon = True
        self.image_extractor_thread.start()

    def get_experiment_list(self, data_folder):                                                            #quand on appelle on donne le data_folder, self créé automatiquement car class
        res = {}
        needed_files = [s + '.csv' for s in self.needed_sensors] + ['rgb.avi', 'depth.avi']                #[camera.csv, emg.csv] [rgb.avi, depth.avi]
        exp_list = []
        exp_list = [x[0] for x in os.walk(data_folder) if all(item in x[2] for item in needed_files)]      #generate file names in the directory  
        #x prend la valeur des files name, item prend valeurs needed files, si item est dans x[2] alors true, si tous les items sont bien dans x[2] alors exp_list est x[0]
        exp_list.sort()
        for exp in exp_list:
            res[exp] = exp   #cree structure pour pouvoir ranger nos valeurs dedans ensuite
        return res
        #donne la liste des experiences pour lesquelles on a bien les fichiers qu'il nous faut (camera et emg et rgb et depth)
        
    def get_indexes(self, initial_guesses, idx):
        max_index = self.get_nb_frames()                         #appelle la fonction dans la meme classe get_nb_frames: returns lenght of self.data in column "camera" = nb total d'images (besoin data remplie avant par une autre fonction)
        camera_index = initial_guesses['camera'][idx]            #prend element idx dans la dimension camera dans initial guesses 
        camera_frame = self.data['camera'].iloc[camera_index]    #prend la frame qui correspond a l'index
        time = camera_frame[2]                                   #donne la ref de temps de la frame
        # extract the other sensor data (exept camera)
        indexes = {}
        indexes['camera'] = camera_index                         #crée nvlle structure pour ranger les camera index
        for s in self.sensor_list[1:]:                           #list créée dans fonction play
            index = initial_guesses[s][idx]                      #prend index idx pour chaque sensor dans data initial guess
            frame = self.data[s].iloc[index]                     #prend la valeur pour le sensor s a la position indexée
            indexes[s] = index                                   #range notre index
            prev_sign = np.sign(time - frame[2])                 #donne -1, 0 ou 1 selon le signe de la diff du current time par rapport au temps de la frame du nvl index
            while abs(time - frame[2]) > 1e-3:                   #tant que la difference entre les 2 est non negligeable
                sign = np.sign(time - frame[2])
                if prev_sign - sign != 0:                        #si signe devient different du prev_sign c'est qu'on est passé devant/derriere la frame de reference = on arrete 
                    break
                index = int(index + sign * 1)                    #on ajoute 1 si sign=1 = time est plus grand que frame[2] = fram[2] est "derrière" la frame de reference
                                                                 #on enlève 1 si frame[2] est "devant" la frame de ref = on se rapproche de la frame de ref
                if index < 0 or index >= self.get_nb_sensor_frames(s):   #on sort si les index sortent de leurs vlauers possibles
                    break
                frame = self.data[s].iloc[index]                 #frame du nvl index
                indexes[s] = index                               #stock  index pour chaque sensor = on garde slmt le dernier donc le plus proche 
                prev_sign = sign
        return indexes
    #Donne les index pour tous les sensors correspondant à un moment de la vidéo (?) (intial guess et idx: index des moments dont on veut les index pour les autres ensors ?)

    def get_window_data(self, indexes):
        start_frame = indexes['camera'][0]                  #premier element de indexes pour camera
        stop_frame = indexes['camera'][1]                   #deuxieme
        max_index = self.get_nb_frames()                    #total number of frame = length of data of camera
        if start_frame < 0 or start_frame > max_index:      #si la start frame est invalide
            print('Incorrect index, expected number between 0 and ' + str(max_index) + ' got ' + str(start_frame))
            return -1
        if stop_frame < 0 or stop_frame > max_index:        #si stop frame est invalide
            print('Incorrect index, expected number between 0 and ' + str(max_index) + ' got ' + str(stop_frame))
            return -1
        start_indexes = self.get_indexes(indexes, 0)        #utilise la fonction d'avant pour avoir les indexes des mesures les plus proches de start frame et stop frame
        stop_indexes = self.get_indexes(indexes, 1)
        window_data = {}
        for s in self.sensor_list:
            window_data[s] = self.data[s].iloc[start_indexes[s]:stop_indexes[s]+1]  
            #stock dans window data pour chawue sensor les valeurs des index dans la partie qui nous interesse (start a stop compris)
        return window_data

    def get_nb_frames(self):
        return len(self.data['camera'])        #longueur du "vecteur" de données de la caméras = nb de frames

    def get_nb_sensor_frames(self, sensor):
        return self.data[sensor].count()[0]    #count sans valeur à chercher ? donne juste le nombre d'élément ?

    def init_image_list(self):
        for t in ['rgb', 'depth']:
            self.images[t] = []
            for i in range(self.get_nb_frames()):          #i prend les valeurs de 1 a nb de frames
                self.images[t].append(self.blank_image)    #pour rgb ou depth dans la structure, on ajoute a la suite une blank image
        #on crée un suite d'image vide pour rgb et avi pour chaque frame

    def extract_images(self):
        def extract_image(video):
            _, frame = video.read()                        #ignore 1st value of video.read
            _, buffer = cv2.imencode('.jpg', frame)        #convert image format into streaming data = compression, current format is jpg
            return buffer

        while True:
            if self.data_changed:                          #if self_data_changed is true
                self.data_changed = False                  #on le passe a faux
                rgb_video = cv2.VideoCapture(join(self.exp_folder, 'rgb.avi'))           #open video file
                depth_video = cv2.VideoCapture(join(self.exp_folder, 'depth.avi'))
                for i in range(self.get_nb_frames()):                 #i de 0 au nb de frame
                    rgb_image = extract_image(rgb_video)              #fonction pour extraire image de la video
                    depth_image = extract_image(depth_video)
                    with self.mutex:                       #??????????????????? LOCK()
                        if self.data_changed:              #break quand remise a true
                            break
                        self.images['rgb'][i] = rgb_image          #on stock les images
                        self.images['depth'][i] = depth_image
                rgb_video.release()       #used to release an acquired lock
                depth_video.release()
            time.sleep(0.01)

    def get_image(self, video_type, frame_index):
        max_index = self.get_nb_frames()
        if frame_index < 0 or frame_index > max_index:   #erreur si frame index incorrect
            print('Incorrect index, expected number between 0 and ' + str(max_index) + ' got ' + str(frame_index))
            return -1
        with self.mutex:          #?????????????????
            image = self.images[video_type][frame_index]   #prend image stockée pour la video type qu'on veut et l'index i (rgb ou depth, et index de position) (stock créé avec fonction d'avant)
        return image

    def align_relative_time(self):
        min_time = max([self.data[s]['relative_time'].iloc[0] for s in self.sensor_list])   #s prend la valeur des sensors, on prend la 1ere valeur de relative time dans les data, et on prend le max comme temps minimum
        max_time = min([self.data[s]['relative_time'].iloc[-1] for s in self.sensor_list])  #on prend les dernières valeurs pour chaque sensor et on prend le min de toutes ces valeurs

        for s in self.sensor_list:          #pour chaque sensor
            self.data[s] = self.data[s][np.logical_and(self.data[s]['relative_time'] - min_time > 1e-3, self.data[s]['relative_time'] - max_time < 1e-3)] #true si la difference entre le relative time et le min time est non négligeable ET la difference entre le rel time et le max time est négative ou négligeable (sert à quoi ???)
            self.data[s]['relative_time'] -= min_time    #self.data = self.data - min_time: on soustrait a la valeur le min_time
#POUR AVOIR LE RELATIEV TIME DU SENSOR QUI A COMMENCE EN DERNIER A 0, ET POUR LES AUTRES SENSORS ON A RELATIVE TIME QUI COMMENCE EN NEGATIFE, COMME CA ON A DES VALEURS POUR TOUS LES SENSORS SI ON SE PLACE A UN RELATIVE TIME POSITIF 
            
        start_camera_index = self.data['camera'].index[0]-1     #1er index moins 1
        stop_camera_index = self.data['camera'].index[-1]       #dernier index
        self.offset = start_camera_index
        for image in ['rgb', 'depth']:                          #pour les 2 types de video
            self.images[image] = self.images[image][start_camera_index:stop_camera_index]    #on remplace dans image les valeurs entre start et stop = valeurs début et fin de la camera

    def play(self, exp_folder):
        self.exp_folder = exp_folder        #folder ou on va chercher les data
        with self.mutex:                    #?????
            indexes = {}
            self.sensor_list = []
            for s in self.available_sensors:     #s prend les valeurs de tous les sensor possibles
                sensor_datafile = join(exp_folder, s + '.csv')       
                if exists(sensor_datafile):
                    self.sensor_list.append(s)                           #si un .csv existe pou le sensor alors on l'ajoute a la liste
                    self.data[s] = pd.read_csv(sensor_datafile)          # et on lit le csv pour l'ajouter a la structure des data
            self.init_image_list()                               #appelle les fonctions 
            self.align_relative_time()
            self.data_changed = True
        
    def export(self, folder, indexes):
        export_folder = join(self.exp_folder, folder)   #crée path vers ls folder
        if not os.path.exists(export_folder):           #si existe pas
            os.makedirs(export_folder)                  #créé les directory
        start_index = indexes['camera'][0]
        stop_index = indexes['camera'][1]
        cut_camera_data = self.data['camera'].iloc[start_index:stop_index+1]   #prend les data de start a stop donné par la variable indexes
        cut_camera_data.to_csv(join(export_folder, 'camera.csv'))              #exporte le fichier csv pour les indexes spécifiés 
        print('Camera data file exported')
        cut_data = self.get_window_data(indexes)                               #get the data for the indexes specified
        # export all csv
        for key, value in cut_data.items():
            value.to_csv(join(export_folder, key + '.csv'), index=False)               #exporte en csv pour les indexes specifies pour chaque sensor
            print(key + ' data file exported')
        # copy calibration file if it exists
        for i in range(1, 3):
            calibration_file = 'FingerTPS_EPFL' + str(i) + '-cal.txt'
            if os.path.exists(join(self.exp_folder, calibration_file)):               #si le calibration file existe
                print('calibration file ' + calibration_file + ' exported')
                copyfile(join(self.exp_folder, calibration_file), join(export_folder, calibration_file))  #on le copie dans nouveau folder avec les fichiers tronqués
        self.export_video(export_folder, start_index, stop_index)                     #exporte les videos tronquées
        print('Export complete')
        
        #POUR EXPORTER TOUS LES FICHIERS AVEC SLMT UN CERTAIN BOUT DE DONNEES

    def export_video(self, folder, start_index, stop_index):
        for t in ['rgb', 'depth']:
            original_video = cv2.VideoCapture(join(self.exp_folder, t + '.avi'))
            cut_video = cv2.VideoWriter(join(folder, t + '.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)
            for i in range(self.offset + stop_index + 1):
                _, frame = original_video.read()
                if i < self.offset + start_index:
                    continue
                cut_video.write(frame)
            print(t + ' video file exported')
            original_video.release()
            cut_video.release()
            
        #TO EXPORT ONLY ONE PART OF THE VIDEO, FROM START INDEX TO STOP INDEX