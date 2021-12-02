# DeepLabv3 in TensorFlow 2

Generate a TensorFlow 2 SavedModel for person semantic segmentation
using DeepLabv3.

Based on code from https://github.com/tensorflow/models/tree/master/research/deeplab, but packaged as a TF2
SavedModel, not a TF1-style frozen graph.

## Usage

Export the model with:
```bash
./export_it.sh ./deeplabv3-pascal-person
```

Then use the exported model as:
```python
import tensorflow as tf
model = tf.saved_model.load('./deeplabv3-pascal-person')
images = tf.zeros([batch_size, any_height, any_width, 3], tf.uint8)  # RGB image batch

person_probability_maps = model(images)
# result is a Tensor with shape [batch_size, any_height, any_width] and dtype tf.float32
# contain per pixel the probability that it's a person pixel
