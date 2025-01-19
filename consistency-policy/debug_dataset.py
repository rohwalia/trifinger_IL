import h5py
import numpy as np
import shutil
import matplotlib.pyplot as plt

# Paths
hdf5_path = "/home/local_rohan/consistency-policy/data/real_lift/real_lift.hdf5"
output_path = "/home/local_rohan/consistency-policy/data/real_lift/real_lift_clip.hdf5"

# Action clipping bounds
n_fingers = 3  # Number of fingers (modify if needed)
thresh = np.pi
low = np.array([-thresh, -thresh, -thresh] * n_fingers, dtype=np.float32)
high = np.array([thresh, thresh, thresh] * n_fingers, dtype=np.float32)

# Create a backup of the original file
shutil.copy(hdf5_path, output_path)

# Open the copied file for modifications
clipped_any = False  # Flag to check if any values are clipped
with h5py.File(output_path, 'a') as f:
    demo_keys = list(f.keys())  # Get all demo keys
    
    # Step 1: Clip action values
    for demo_key in demo_keys:
        if demo_key.startswith('demo_'):
            action_data = f[demo_key]['action'][:]  # Read the action array
            
            # Identify which values will be clipped
            clipped_mask = (action_data < low) | (action_data > high)
            if np.any(clipped_mask):  # Check if any values are actually clipped
                clipped_any = True
                print(f"Values clipped in {demo_key}")
                
                # Clip values
                clipped_action = np.clip(action_data, low, high)
                f[demo_key]['action'][...] = clipped_action  # Write back to the file
    
    if not clipped_any:
        print("No values were clipped in any demos.")
    
    # Step 2: Combine all clipped action arrays for histogram
    all_actions = []
    for demo_key in f.keys():
        if demo_key.startswith('demo_'):
            action_data = f[demo_key]['action'][:]
            all_actions.append(action_data)
    all_actions = np.vstack(all_actions)  # Stack all actions into a single array

    # Step 3: Plot histograms for each action dimension
    num_dims = all_actions.shape[1]  # Assuming shape = (N, 9)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # 3x3 grid for 9 dimensions
    axes = axes.flatten()  # Flatten to loop easily

    for i in range(num_dims):
        axes[i].hist(all_actions[:, i], bins=50, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Action Dimension {i+1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

print(f"\nClipped HDF5 file saved at: {output_path}")
