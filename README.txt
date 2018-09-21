This is the code for reproducing experiments in the paper, "Learning 6-DoF Grasping and Pick-Place Using Attention Focus".

Prerequisites: OpenRAVE, Caffe, 3DNet models, and Matlab with the Python interface (for GPD comparison).

Step 1: Run python/test_models_*.py to generate the model files.
Step 2: Run python/train_pick_and_place_*.py for joint training of pick and place.
Step 3: Run python/train_grasping_*.py for the training grasping only.
Step 4: Run python/test_grasping_*.py for testing the trained grasp agent.
Step 5: Run python/test_gpd.py to test GPD in the OpenRAVE environment.

Files of interest:
1) Reward functions are in the rl_environment_*.py files.
2) HSE3S is implemented in the rl_agent files.
3) The main training files are train_pick_and_place_*.py.
4) The "antipodal" condition is defined in rl_environment_grasping.IsAntipodalGrasp.
