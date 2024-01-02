# ViNT


## Training for Language Specified Subgoals
Language Encoder uses CLIP, the rest of the weights are loaded from `vint.pth`
Check the `config/langvint.yaml` to see where to place the `vint.pth` (should be renamed to `latest.pth`
Modify the dataset paths in the config file according to where you placed them

Make sure to set `goal_input_type: text` in the config file.

`python train.py -c config/langvint.yaml`
