# OPEN TEACH: A Versatile Teleoperation System for Robotic Manipulation
## Setting up code
Clone this repo and create a conda env with this command:\
```conda env create -f environment.yaml```\
Might be a couple of packages that are not included, just install those when trying to run the code.\
Also install the library on the rohan_dev branch of this repo: https://gitlab.is.tue.mpg.de/nguertler/trifingermujocoenv/-/tree/rohan_dev?ref_type=heads\
Activate the conda env by:\
```conda activate openteach```\
and then use this command:\
```pip install git+ssh://git@gitlab.is.tue.mpg.de/nguertler/trifingermujocoenv.git@rohan_dev#egg=trifinger_mujoco_env```\
WARNING: when the branch receives new commits, you have to upgrade the library manaully to get it on your system

## Setting up trifinger network to connect the MetaQuest3 wirelessly
For your system to recognize the BrosTrend Wifi adapter, install this driver: http://linux.brostrend.com/ \
Execute the ```setup_trifinger.sh``` bash file: \
```chmod +x setup_trifinger.sh```\
```bash setup_trifinger.sh```\
This bash file creates a wireless network with ssid=trifinger and password=mpirobotgang \
Info on the single steps can be found here: https://arnab-k.medium.com/ubuntu-how-to-setup-a-wi-fi-hotspot-access-point-mode-192cbb2eeb90

To start the network, execute the following commands: \
```sudo service hostapd restart```\
```sudo service isc-dhcp-server start```\
This has to be done everytime you restart your system. Make sure the BrosTrend Wifi Adapter is connected to your system.\
IMPORTANT: the variable INTERFACE inside the bash file is specific to the BrosTrend device

## Setting up software on MetaQuest3
Follow the installation steps for the Multi Robot Arm (Bimanual) given in the original OpenTeach repo: https://github.com/aadhithya14/Open-Teach/blob/main/docs/vr.md\
If you are using the trifinger network with the specs from the bash file, enter the IP address ```10.10.0.1``` when starting the application.

## Running code and workflow
Now everything should be installed on the MetaQuest3 and your system. Make sure the MetaQuest3 is connected to the trifinger network hosted by your system. Enter the VR application with the Unity logo. Now you should see a blank white screen. Then on your system execute this example command in terminal with the openteach conda env activated:\
```python3 teleop.py robot=trifinger_sim sim_env=True collect=True task=push_reorient```\
There are several flags you can set:
- ```sim_env=True``` disables components of OpenTeach associated with sensors and cameras
- ```collect=True``` enables the recording of demonstrations (will be saved in a directory ```extracted_data/```)
- ```task=push_reorient``` defines for which task the data is collected, if task=none then there is no goal position
- ```user=Max``` sets the demonstrator for data collection and later a leaderboard will be created for people who recorded the best data (if you want to be on there make sure to define a user when recording data)

Now you should be able to see a plot in the left upper corner showing the tracked keypoints of your right hand. To start teleoperation you have initialize the simulation via hand gestures. In our configuration three finger tips are tracked: left index, right thumb and right index. Arrange these fingers somewhat like in a triangle. Important is only that the left index finger tip is most left and the right index finger tip is furthest away from you. Once this is the case briefly touch your right thumb and right middle finger. Now you should see the simulation and markers mimicking your finger tip positions and setting the target for the trifinger tips. IMPORTANT: if your right thumb and right middle finger touch again the simulation will stop

To record demonstrations, your left hand is needed. To start recording touch your left thumb and left index finger. The border of the window into the simulation should turn green. To stop recording, touch your left thumb and left middle finger. Now the border is red again. Whenever stopping a recording the task is reset where the goal cube and actual cube get a new position/orientation. IMPORTANT: It always takes a couple of seconds for the data of a run to be saved, so don't start the next recording immediately.

## Troubleshooting

If you get some error with OpenGL, might wanna run this:\
```export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6```

If you get an error when trying to run the ```setup_trifinger.bash``` file and it stems from ```/etc/network/interfaces``` then this might solve it: https://askubuntu.com/questions/1276847/looking-for-etc-network-interfaces-is-missing

The network configuration we define allows the BrosTrend Wifi adapter to only assign one IP address. You might get an error with the isc-dhcp-server if you try to connect a second device. This is because the server stores preivous connections in ```/var/lib/dhcp/dhcpd.leases```. So just delete the lease with 10.10.0.2 in this file and the issue should be resolved.


