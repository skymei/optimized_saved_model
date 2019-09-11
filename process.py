from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile


def freeze_model(saved_model_dir, output_node_names, output_filename):
    output_graph_filename = os.path.join(saved_model_dir, output_filename)
    initializer_nodes = ''
    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_filename,
        saved_model_tags=tag_constants.SERVING,
        output_node_names=output_node_names,
        initializer_nodes=initializer_nodes,
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name="save/restore_all",
        filename_tensor_name="save/Const:0",
        clear_devices=True,
        input_meta_graph=False,
    )
    print('graph freezed!')


def load_graph_def(filename):
    input_graph_def = tf.GraphDef()
    with gfile.FastGFile(filename, 'rb') as file:
        data = file.read()
        input_graph_def.ParseFromString(data)
    return input_graph_def


def optimize_graph(base_path, freezed_name):
    frozen_graph_filename = os.path.join(base_path, freezed_name)
    graph_def = load_graph_def(frozen_graph_filename)

    optimized_graph = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def=graph_def,
        input_node_names=['image'],
        placeholder_type_enum=dtypes.float32.as_datatype_enum,
        output_node_names=['deprocess_stylized_image_1/mul'],
        toco_compatible=True
    )

    optimized_graph_filename = os.path.join(base_path, 'optimized_model.pb')
    tf.train.write_graph(
        optimized_graph, '', optimized_graph_filename, as_text=False
    )
    print('graph optimized!')


def convert_graph_def_to_saved_model(export_dir, graph_filepath):
    if tf.gfile.Exists(export_dir):
        tf.gfile.DeleteRecursively(export_dir)
    graph_def = load_graph_def(graph_filepath)
    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name='')
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs={node.name: session.graph.get_tensor_by_name(
                '{}:0'.format(node.name))
                for node in graph_def.node if node.op == 'Placeholder'},
            outputs={'output_image': session.graph.get_tensor_by_name(
                'deprocess_stylized_image_1/mul:0')}
        )
        print('Optimized graph converted to SavedModel!')


freeze_model('single/1', 'deprocess_stylized_image_1/mul', 'freezed_model.pb')
optimize_graph('single/1', 'freezed_model.pb')
convert_graph_def_to_saved_model('single_optimized/1', 'single/1/optimized_model.pb')
