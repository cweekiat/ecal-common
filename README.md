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

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/cweekiat/ecal-common.git
   cd ecal-common

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Running Detection Program:**
   ```bash
   cd ecal-common/python/
   python3 test_detection3d.py
   ```



3. **OLD: Running Detection Program:**

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

[Watch Demo 1](python/assets/demo.mp4)

[Watch Demo 2](python/assets/demo_depth.mp4)

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code for both non-commercial and commercial purposes.

Happy coding!
