import keras
from keras.models import load_model

from tensorflow.python.framework import graph_util

import tensorflow as tf
sess = tf.Session()

import keras.backend as K
K.set_learning_phase(0)   # avoid output keras_learning_phase placeholder :)
K.set_session(sess)


model = load_model("mnist_cnn.h5")
#print(model.input, model.output)

input_tensor = model.input.name.split(":")[0]
output_tensor = model.output.name.split(":")[0]

output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_tensor])
tf.train.write_graph(output_graph_def, '../../model-zoo/tf/mnist', 'keras-tf-mnist.pb', False)

print "\nInput Tensor: ", input_tensor
print "Output Tensor:", output_tensor
