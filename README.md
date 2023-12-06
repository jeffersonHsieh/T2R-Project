# T2R-Project

# install dependencies
```
conda env create -f train_environment.yml
pip install -e visualnav-transformer/train
pip install -e visualnav-transformer/diffusion_policy
pip install airsim
pip install pynput
```
For Windows, you should install these in WSL2.

# Airsim Setup
We tested this setup on Ubuntu 20.04
1. Build Unreal Engine 4.27 from github src
2. Install [ROS-noetic](https://wiki.ros.org/noetic/Installation/Ubuntu). Then install tf2 sensor and mavros packages: `sudo apt-get install ros-noetic-tf2-sensor-msgs ros-noetic-tf2-geometry-msgs ros-noetic-mavros*`. For Windows, you would do this in WSL2.
3. Build Airsim from src
```
git clone https://github.com/Microsoft/AirSim.git;
cd AirSim;
./setup.sh;
./build.sh;
```
For Windows you need to build this in WSL2

4. Build Airsim RosBridge using Catkin
For Windows, you would do the following steps in WSL2.
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
# the wasd key cannot control the vehicle! only external camera. Computer Vision mode also doesn't work as it only controls the external camera!
# to control the vehicle without a remote control, you can use the python api (see the test setup section). 

~/Downloads/AirSimNH/LinuxNoEditor/AirSimNH/Binaries/Linux/AirSimNH -settings=$(realpath ReasonedExplorer/src/settings_linux.json)
```
In Windows it would be a similar command.
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
If you're on a Windows system, you should follow [this section](https://microsoft.github.io/AirSim/airsim_ros_pkgs/#setting-up-the-build-environment-on-windows10-using-wsl1-or-wsl2). Specifically, you would need to use a port following [this](https://microsoft.github.io/AirSim/airsim_ros_pkgs/#how-to-run-airsim-on-windows-and-ros-wrapper-on-wsl)


Alternatively you can also directly interface with the [python client](https://microsoft.github.io/AirSim/api_docs/html/#)


Now, you can start interacting with the environment.
ROS example
```
cd tests
python test_airsim_ros.py
```

For WSL, you might also want to change the settings.json by adding `"LocalHostIp": "YOUR WSL_HOST_IP",`
You can get it by `export WSL_HOST_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')`

Python Interactive Shell
```
import airsim
HOST = '127.0.0.1' # Standard loopback interface address (localhost)
from platform import uname
import os
if 'linux' in uname().system.lower() and 'microsoft' in uname().release.lower(): # In WSL2
    if 'WSL_HOST_IP' in os.environ:
        HOST = os.environ['WSL_HOST_IP']
        print("Using WSL2 Host IP address: ", HOST)
client = airsim.MultirotorClient(ip=HOST)
client.confirmConnection()
client.reset() # should be called before enabling ApiControl
client.enableApiControl(True)
client.takeoffAsync().join()
# airsim uses NED axis
client.moveToPositionAsync(60, -40, -20, 5, timeout_sec=10).join() # move to world frame coordinate in NED
client.moveByVelocityAsync(1,0,-1,duration=10).join()
```


# Training Data Collection
## ROS-Bags
There are two ways to collect the rosbag trajectories. One is to use WASD control, the other is use nomad for exploration and use WASD to override in case of collision, since the simulation environment is out of distribution.

```
cd visualnav-transformer/deployment/src
mkdir ../model_weights
```

Remember to set the correct path to the rosbridge setup script you've built in these scripts
```
ros_bridge_setup=/path/to/AirSim/ros/devel/setup.bash
```
If you're on windows, uncomment these lines in the script and comment out the original roslaunch
```
export WSL_HOST_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
roslaunch airsim_ros_pkgs airsim_node.launch output:=screen host:=$WSL_HOST_IP
```


Only WASD. This script starts 3 tmux panes, the ros bridge and the keyboard controller would get started automatically

But you would need to manually press "ENTER" to start recording. Once you're done, interrupt the ros record command.
The bags will be saved under `visualnav-transformer/deployment/topomaps/bags`
If you're on windows comment this section out if you're using WSL, launch the keyboard_sim_controller.py in Windows OS not WSL
```
./record_bag.sh airsim_test

```
