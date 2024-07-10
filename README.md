# eCAL Common Headers and Definitions

This repository includes schema definitions for serialization.

## 3D Drone Detection Model

This repository contains a 3D drone detection model designed for target drone detection, tracking, position estimation, and velocity estimation.

![](python/assets/demo_gif.gif)

## Features

The code in this repository provides the following functionalities:

1. **Target Drone Detection:**
   - Utilizes advanced computer vision techniques and deep learning to identify and locate drones in a 3D space.

2. **Target Drone Tracking:**
   - Implements tracking algorithms to follow the movement of identified drones over time.

3. **Target Drone Position Estimation:**
   - Utilizes the detected features to estimate the precise position of the target drone in a 3D coordinate system.

4. **Target Drone Velocity Estimation:**
   - Calculates the velocity of the target drone based on its movement patterns.

## Usage

Follow the steps below to use the 3D drone detection model:

1. **Install MiniConda3**
     - Download Miniconda package [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh)
   ```bash
   cd ~/Downloads
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

3. **Install Ecal**
   In a new terminal
   ```bash
   sudo add-apt-repository ppa:ecal/ecal-5.12
   sudo apt-get update
   sudo apt-get install ecal
   
   sudo apt install python3-ecal5
   ```
5. **Create Conda Environment**
   ```bash
   cd ecal-common/python
   conda create -n vilota python=3.10
   ```

6. **Install Dependencies:**
   ```bash
   conda activate vilota
   pip install matplotlib==3.5.1 numpy==1.25.0 pycapnp==1.3.0 ultralytics==8.0.220 protobuf
   ```

7. **Running Detection Program:**
   ```bash
   python3 test_detection3d.py
   ```
8. **If you face this error**
   ```bash
   import ecal.core.core as ecal_core
   ModuleNotFoundError: No module named 'ecal'
   ```
   1. Open test_detection3.py
   2. Add these lines to the top of the python file
      ```bash
      import sys
      print(sys.path)
      sys.path.append('/usr/local/lib/python3.8/dist-packages')
      ```


<!--
4. **OLD: Running Detection Program:**

  3.1 In the first terminal, SSH into Vilota Camera and run:
   ```bash
   vk_camera_driver ~/vilota_configs_common/camera_driver/vk360_light_front_rectified.json
   ```
  3.2 In the second terminal, SSH into Vilota Camera and run:
   ```bash
   vk_vio_ecal ~/vilota_configs_common/vio/vk180_moderate_rectified.json
   ```
  3.2 In the third terminal, run:
   ```bash
   python3 test_detection3d.py
   ```
-->
[Watch Demo 1](python/assets/demo.mp4)

[Watch Demo 2](python/assets/demo_depth.mp4)

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code for both non-commercial and commercial purposes.

Happy coding!
