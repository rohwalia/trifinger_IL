from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax

block = ResNet18Conv(input_channel=3, input_coord_conv=False)
print(block.nets)
pool = SpatialSoftmax(input_shape=[3, 260, 350], num_kp=32, temperature=1.0, noise_std=0.0)
print(pool)