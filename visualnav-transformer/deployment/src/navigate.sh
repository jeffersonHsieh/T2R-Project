#!/bin/bash

# Create a new tmux session
session_name="vint_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# create new tmux window for roslaunch
tmux new-window -n "cmd_vel_mux"
# split the window into two panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "source ${ros_bridge_setup}" Enter
tmux send-keys "roslaunch airsim_ros_pkgs airsim_node.launch" Enter

# sleep for 5 seconds
sleep 5

# run the airsim adapter
tmux select-pane -t 1
tmux send-keys "conda activate t2r_vint" Enter
tmux send-keys "python airsim_adapter.py" Enter

# switch back to the first window
tmux select-window -t 0


# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves

# Run the navigate.py script with command line args in the first pane
tmux select-pane -t 0
tmux send-keys "conda activate t2r_vint" Enter
tmux send-keys "python navigate_airsim.py $@" Enter

# Run the teleop.py script in the second pane
tmux select-pane -t 1
tmux send-keys "conda activate t2r_vint" Enter
tmux send-keys "python keyboard_vel_mux_controller.py" Enter


# Run the pd_controller.py script in the third pane
tmux select-pane -t 2
tmux send-keys "conda activate t2r_vint" Enter
tmux send-keys "python pd_controller.py" Enter


# Run the vint_airsim_mux.launch in the fourth pane
tmux select-pane -t 3
tmux send-keys "source ${ros_bridge_setup}" Enter
tmux send-keys "roslaunch vint_airsim_mux.launch" Enter


# Attach to the tmux session
tmux -2 attach-session -t $session_name
