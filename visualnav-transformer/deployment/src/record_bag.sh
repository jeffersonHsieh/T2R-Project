#!/bin/bash

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "source ~/Desktop/AirSim/ros/devel/setup.bash" Enter
tmux send-keys "roslaunch airsim_ros_pkgs airsim_node.launch" Enter

# Run the teleop.py script in the second pane
tmux select-pane -t 1
tmux send-keys "conda activate t2r_vint" Enter
tmux send-keys "python keyboard_sim_controller.py" Enter

# Change the directory to ../topomaps/bags and run the rosbag record command in the third pane
tmux select-pane -t 2
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "rosbag record /airsim_node/SimpleFlight/front_center/Scene/compressed -o $1" # change topic if necessary
# tmux send-keys "rosbag record /airsim_node/SimpleFlight/front_center/Scene/compressed /airsim_node/SimpleFlight/right_camera/Scene/compressed /airsim_node/SimpleFlight/left_camera/Scene/compressed -o $1"

# Attach to the tmux session
tmux -2 attach-session -t $session_name