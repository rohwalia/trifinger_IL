# Learning Three-Fingered Manipulation Through Low-Cost Teleoperation and Imitation Learning

[Rohan Walia]()<sup>1</sup>

<sup>1</sup>Technical University of Munich


<img src="overview.png" alt="TriFinger" width="60%"/>

## Codebase structure
The teleoperation implementation is given in the directory [open_teach](./open_teach) and the MuJoCo simulation environment used is available in [trifingermujocoenv](./trifingermujocoenv). For the former, there are instructions for how to setup the VR app on the Meta Quest 3 and the code on the control PC. There are demos with which the simulation environment can be tested. Regarding the imitation learning algorithms, training and evaluation was done for Diffusion Policy (DP) and Consistency Policy (CP) with the [consistency-policy](./consistency-policy) directory and for Explicit Behavior Cloning (EBC) and Action Chunking with Transformers (ACT) with the [act](./act) directory. These are directly taken from the codebases of the respective papers and their README files should give adequate guidance on how to run the algorithms. Important to note, is that EBC is refereed to as BC-ConvMLP in the act directory.

## Videos
