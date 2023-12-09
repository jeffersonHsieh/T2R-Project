#!/bin/bash

# Create a new tmux session
session_name="vint_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "source /mnt/d/airsim/AirSim/ros/devel/setup.bash" Enter
tmux send-keys "roslaunch airsim_ros_pkgs airsim_node.launch host:=$WSL_HOST_IP" Enter

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 1
tmux send-keys "conda activate t2r_vint" Enter
# pass args except the first one (which is the name of the rosbag
tmux send-keys "python explore_airsim.py ${@:2}" Enter

# Run the teleop.py script in the third pane
tmux select-pane -t 2
# tmux send-keys "source /mnt/d/airsim/AirSim/ros/devel/setup.bash" Enter
# tmux send-keys "roslaunch airsim_ros_pkgs position_controller_simple.launch output:=screen host:=$WSL_HOST_IP"
tmux send-keys "conda activate t2r_vint" Enter
# tmux send-keys "python keyboard_sim_controller.py"

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 3
tmux send-keys "conda activate t2r_vint" Enter
tmux send-keys "python pd_controller.py" Enter


tmux select-pane -t 4
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "rosbag record /airsim_node/SimpleFlight/front_center/Scene/compressed \
/airsim_node/SimpleFlight/imu/imu \
/airsim_node/SimpleFlight/odom_local_ned \
/airsim_node/SimpleFlight/gps/gps \
/airsim_node/SimpleFlight/global_gps \
/airsim_node/SimpleFlight/left_camera/Scene/compressed \
/airsim_node/SimpleFlight/right_camera/Scene/compressed -o $1" # change topic if necessary


# Attach to the tmux session
tmux -2 attach-session -t $session_name
