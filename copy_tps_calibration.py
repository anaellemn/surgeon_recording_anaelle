from os.path import join
import os
import time
from shutil import copyfile

if __name__ == "__main__":
    calibration_dir = r'C:\Program Files\Pressure Profile Systems\Chameleon TVR\setup'
    destination_dir = join('source', 'surgeon_recording', 'config')
    
    for i in range(1, 3):
        calibration_file = 'FingerTPS_EPFL' + str(i) + '-cal.txt'
        # first remove old calibration files
        if os.path.exists(join(destination_dir, calibration_file)):
            os.remove(join(destination_dir, calibration_file))

        if time.time() - os.path.getmtime(join(calibration_dir, calibration_file)) < 600: # file not older than 10 minutes
            copyfile(join(calibration_dir, calibration_file), join(destination_dir, calibration_file))
            print(calibration_file + ' copied')