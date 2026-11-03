import threading
import time

import ui
from lidar_processor import LidarProcessor

def main():    
    lidar_processor = LidarProcessor("COM3")
    lidar_processor.start()
    ui.run_gui(lidar_processor) #blocks main thread, starts web server and opens browser




if __name__ == "__main__":
    main()
