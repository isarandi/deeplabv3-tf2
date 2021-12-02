# Copyright
# 2018 The TensorFlow Authors All Rights Reserved.
# 2021 Istvan Sarandi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exports trained model to TensorFlow frozen graph."""

import os
import shutil

import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from tensorflow.python.tools import freeze_graph

import common
import model

flags.DEFINE_string('export_path', None, 'Path to output Tensorflow 2 SavedModel.')
flags.DEFINE_integer('num_classes', 21, 'Number of classes.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for inference.')
flags.DEFINE_multi_integer('crop_size', [513, 513], 'Crop size [height, width].')
# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None, 'Atrous rates for atrous spatial pyramid pooling.')
flags.DEFINE_integer('output_stride', 8, 'The ratio of input to output spatial resolution.')
# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale inference.
flags.DEFINE_multi_float('inference_scales', [1.0], 'The scales to resize images for inference.')
flags.DEFINE_bool('add_flipped_images', False, 'Add flipped images during inference or not.')
# Input name of the exported model.
_INPUT_NAME = 'ImageTensor'
# Output name of the exported model.
_OUTPUT_NAME = 'SemanticPredictions'


def main(unused_argv):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    zippath = tf.keras.utils.get_file(
        origin='http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        extract=True, cache_subdir='models')
    checkpoint_path = os.path.dirname(zippath) + '/deeplabv3_pascal_trainval/model.ckpt'
    frozen_graph_path = '/tmp/deeplabv3.pb'

    save_frozen_graph(checkpoint_path, frozen_graph_path)
    save_intermediate_tf2_model(frozen_graph_path)
    save_tf2_model()


def save_frozen_graph(checkpoint_path, frozen_graph_path):
    with tf.Graph().as_default():
        image = tf.compat.v1.placeholder(
            tf.uint8, [None, FLAGS.crop_size[0], FLAGS.crop_size[1], 3], name=_INPUT_NAME)

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: FLAGS.num_classes},
            crop_size=FLAGS.crop_size, atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)

        if tuple(FLAGS.inference_scales) == (1.0,):
            predictions = model.predict_labels(
                image, model_options=model_options, image_pyramid=FLAGS.image_pyramid)
        else:
            predictions = model.predict_labels_multi_scale(
                image, model_options=model_options, eval_scales=FLAGS.inference_scales,
                add_flipped_images=FLAGS.add_flipped_images)

        predictions = tf.cast(predictions[common.OUTPUT_TYPE], tf.float32)
        # Crop the valid regions from the predictions.
        semantic_predictions = tf.identity(predictions, name=_OUTPUT_NAME)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.model_variables())
        os.makedirs(os.path.dirname(FLAGS.export_path), exist_ok=True)
        freeze_graph.freeze_graph_with_def_protos(
            tf.compat.v1.get_default_graph().as_graph_def(add_shapes=True),
            saver.as_saver_def(), checkpoint_path, _OUTPUT_NAME, restore_op_name=None,
            filename_tensor_name=None, output_graph=frozen_graph_path, clear_devices=True,
            initializer_nodes=None)


def save_intermediate_tf2_model(frozen_graph_path):
    g = load_graph(frozen_graph_path)
    x = g.get_tensor_by_name('prefix/ImageTensor:0')
    y = g.get_tensor_by_name('prefix/SemanticPredictions:0')
    sm = tf.compat.v1.saved_model
    with tf.compat.v1.Session(graph=g) as sess:
        signature_def = sm.signature_def_utils.predict_signature_def(
            inputs=dict(image=x), outputs=dict(mask=y))
        if os.path.exists(FLAGS.export_path):
            shutil.rmtree(FLAGS.export_path)
        os.mkdir(FLAGS.export_path)
        builder = sm.builder.SavedModelBuilder(FLAGS.export_path)
        builder.add_meta_graph_and_variables(
            sess, ['serve'], signature_def_map=dict(serving_default=signature_def))
        builder.save()


def load_graph(frozen_graph_filename):
    with tf.compat.v1.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.compat.v1.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name='prefix')
    return graph


def save_tf2_model():
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.enable_eager_execution()
    base_model = tf.saved_model.load(FLAGS.export_path)
    shutil.rmtree(FLAGS.export_path)
    segmentation_net = PersonSegmenter(base_model)
    tf.saved_model.save(segmentation_net, FLAGS.export_path)


class PersonSegmenter(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.predict_fn = self.model.signatures['serving_default']

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8)])
    def __call__(self, image):
        """RGB image batch to person semantic segmentation mask"""
        proc_size = 513
        target_size = tf.convert_to_tensor(proc_size, tf.float32)
        shape = tf.shape(image)

        # 1. Resize
        h = tf.cast(shape[1], tf.float32)
        w = tf.cast(shape[2], tf.float32)
        factor = target_size / tf.maximum(h, w)
        target_w = tf.cast(factor * w, tf.int32)
        target_h = tf.cast(factor * h, tf.int32)
        if factor > 1:
            image = tf.image.resize(
                image, (target_h, target_w), method=tf.image.ResizeMethod.BILINEAR)
        else:
            image = tf.image.resize(
                image, (target_h, target_w), method=tf.image.ResizeMethod.AREA)
        # 2. Pad
        rest_h = proc_size - target_h
        rest_w = proc_size - target_w
        image = tf.pad(image, [(0, 0), (rest_h, 0), (rest_w, 0), (0, 0)])
        image = tf.cast(image, tf.uint8)
        # 3. Predict
        mask = self.predict_fn(image=image)['mask']
        # 4. Unpad
        mask = mask[:, rest_h:, rest_w:, tf.newaxis]

        # 5. Resize back
        if factor < 1:
            mask = tf.image.resize(
                mask, (shape[1], shape[2]), method=tf.image.ResizeMethod.BILINEAR)
        else:
            mask = tf.image.resize(
                mask, (shape[1], shape[2]), method=tf.image.ResizeMethod.AREA)
        return tf.squeeze(mask, -1)


if __name__ == '__main__':
    flags.mark_flag_as_required('export_path')
    try:
        app.run(main)
    except SystemExit:
        pass
