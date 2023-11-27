# T2R-Project

# install dependencies
```
conda env create -f train_environment.yml
pip install -e visualnav-transformer/train
pip install -e visualnav-transformer/diffusion_policy
pip install airsim
```

# Airsim Setup
We tested this setup on Ubuntu 20.04
1. Build Unreal Engine 4.27 from github src
2. Install [ROS-noetic](https://wiki.ros.org/noetic/Installation/Ubuntu). Then install tf2 sensor and mavros packages: `sudo apt-get install ros-noetic-tf2-sensor-msgs ros-noetic-tf2-geometry-msgs ros-noetic-mavros*`
3. Build Airsim from src
```
git clone https://github.com/Microsoft/AirSim.git;
cd AirSim;
./setup.sh;
./build.sh;
```
4. Build Airsim RosBridge using Catkin
```
pip install "git+https://github.com/catkin/catkin_tools.git#egg=catkin_tools"
sudo apt update && sudo apt install gcc-8 g++-8
```
under the cloned AirSim root
```
cd ros
catkin build -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 -DPYTHON_EXECUTABLE=/usr/bin/python3
```
5. Download a precompiled airsim environment from [release](https://github.com/microsoft/AirSim/releases/tag/v1.8.1) and unzip it. We tested on [AirSimNH](https://github.com/microsoft/AirSim/releases/download/v1.8.1/AirSimNH.zip)  


## Test Simulation
For the airsim client to have something to interact, we need to first launch the simulation in a separate window/screen
```
# assuming we run this under the root dir of this project and unzipped inside download
unzip ~/Downloads/AirSimNH.zip
# we use the settings_linux.json to set a vehicle/multirotor, otherwise we would only have an external camera
# 

~/Downloads/AirSimNH/LinuxNoEditor/AirSimNH/Binaries/Linux/AirSimNH -settings=$(realpath ReasonedExplorer/src/settings_linux.json)
```
You can press `fn+F11` to toggle full screen mode, and you might need to do this to prevent the window from minimizing if you clicked on other apps
```
echo "export SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS=0" >> ~/.profile
source ~/.profile
```

Launch ROS in a separate screen
```

source devel/setup.bash;
roslaunch airsim_ros_pkgs airsim_node.launch
```
there are several [ROS topics](https://microsoft.github.io/AirSim/airsim_ros_pkgs/#airsim-ros-wrapper-node) airsim is already publishing and subscribing to, we would need to launch the ROS Bridge if we want to interact with the simulation through ROS. 

Alternatively you can also directly interface with the [python client](https://microsoft.github.io/AirSim/api_docs/html/#)


Now, you can start interacting with the environment.
ROS example
```
cd tests
python test_airsim_ros.py
```

Python Interactive Shell
```
import airsim
client=airsim.MultirotorClient()
client.confirmConnection()
client.reset() # should be called before enabling ApiControl
client.enableApiControl(True)
client.takeoffAsync().join()
# airsim uses NED axis
client.moveToPositionAsync(60, -40, -20, 5, timeout_sec=10).join() # move to world frame coordinate in NED
client.moveByVelocityAsync(1,0,-1,duration=10).join()
```

