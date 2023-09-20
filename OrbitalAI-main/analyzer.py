import tensorflow as tf
from bisenet_model import bisenet_v2
from local_utils.config_utils import parse_config_utils

CFG = parse_config_utils.cityscapes_cfg_v2

# Define your model here
input_tensor_size = CFG.AUG.EVAL_CROP_SIZE
input_tensor_size = [int(tmp / 2) for tmp in input_tensor_size]
input_tensor = tf.placeholder(
    dtype=tf.float32,
    shape=[1, input_tensor_size[1], input_tensor_size[0], 3],
    name='input_tensor'
)

model = bisenet_v2.BiseNetV2(phase='test', cfg=CFG)

def create_example_model():
    input_tensor_size = CFG.AUG.EVAL_CROP_SIZE
    input_tensor_size = [int(tmp / 2) for tmp in input_tensor_size]
    input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[1, input_tensor_size[1], input_tensor_size[0], 3],
        name='input_tensor'
    )
    bisenet_model = bisenet_v2.BiseNetV2(phase='test', cfg=CFG)
    prediction = bisenet_model.inference(
        input_tensor=input_tensor,
        name='BiseNetV2',
        reuse=False
    )

    return prediction

# Create the model
model_output = create_example_model()

# Calculate the number of parameters
total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    num_params = 1
    for dim in shape:
        num_params *= dim.value
    total_parameters += num_params

# Calculate the size of the model in bytes
model_size_bytes = total_parameters * 4  # Assuming 32-bit floating point values (4 bytes per parameter)

print(f"Number of Parameters: {total_parameters}")
print(f"Model Size (bytes): {model_size_bytes}")
