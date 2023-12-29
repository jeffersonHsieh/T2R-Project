#!/bin/bash

# Create a new tmux session
session_name="record_bag_$(date +%s)"
ros_bridge_setup=/mnt/d/airsim/AirSim/ros/devel/setup.bash
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves

# Run the roslaunch command in the first pane
tmux select-pane -t 0

tmux send-keys "source ${ros_bridge_setup}" Enter

# comment this and uncomment the next two line to run in wsl
tmux send-keys "roslaunch airsim_ros_pkgs airsim_node.launch" Enter


sleep 5

tmux new-window -n "cmd_vel_mux"
# Run the vint_airsim_mux.launch in the fourth pane
tmux select-pane -t 0
tmux send-keys "source ${ros_bridge_setup}" Enter
tmux send-keys "roslaunch vint_airsim_mux.launch" Enter

tmux select-window -t 0
# tmux send-keys "export WSL_HOST_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')" Enter
# tmux send-keys "roslaunch airsim_ros_pkgs airsim_node.launch output:=screen host:=$WSL_HOST_IP" Enter

# Run the teleop.py script in the second pane
tmux select-pane -t 1
tmux send-keys "conda activate t2r_vint" Enter
tmux send-keys "python airsim_adapter.py" Enter

tmux select-pane -t 2
tmux send-keys "conda activate t2r_vint" Enter
tmux send-keys "python keyboard_vel_mux_controller.py" Enter

# Change the directory to ../topomaps/bags and run the rosbag record command in the fourth pane
tmux select-pane -t 3
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "rosbag record /airsim_node/SimpleFlight/front_center/Scene/compressed \
/airsim_node/SimpleFlight/imu/imu \
/airsim_node/SimpleFlight/odom_local_ned \
/airsim_node/SimpleFlight/gps/gps \
/airsim_node/SimpleFlight/global_gps \
/airsim_node/SimpleFlight/left_camera/Scene/compressed \
/airsim_node/SimpleFlight/right_camera/Scene/compressed -o $1" # change topic if necessary

# /airsim_node/SimpleFlight/front_center/Scene/compressedDepth \
# /airsim_node/SimpleFlight/left_camera/Scene/compressedDepth \
# /airsim_node/SimpleFlight/right_camera/Scene/compressedDepth \


# Attach to the tmux session
tmux -2 attach-session -t $session_name