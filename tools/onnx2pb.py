import numpy as np
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


print('loading onnx model')
onnx_model = onnx.load('/media/yinyunjie/WD_HDD/models/AlexNet/onnx_weights_model/bvlc_alexnet/model.onnx')

print('prepare tf model')

tf_rep = prepare(onnx_model, strict=False)


tf_pb_path = '/media/yinyunjie/WD_HDD/models/AlexNet/onnx_weights_model/bvlc_alexnet/model.pb'
tf_rep.export_graph(tf_pb_path)

with tf.Graph().as_default():
    graph_def = tf.GraphDef()
    with open(tf_pb_path, "rb") as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
    with tf.Session() as sess:
        # init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        # sess.run(init)

        # print all ops, check input/output tensor name.
        # uncomment it if you donnot know io tensor names.
        '''
        print('-------------ops---------------------')
        op = sess.graph.get_operations()
        for m in op:
            print(m.values())
        print('-------------ops done.---------------------')
        '''

        input_x = sess.graph.get_tensor_by_name("data_0:0")  # input
        #outputs1 = sess.graph.get_tensor_by_name('add_1:0')  # 5
        outputs = sess.graph.get_tensor_by_name('Softmax:0')  # 10
        #output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x: img_np})
        output_tf_pb = sess.run([outputs], feed_dict={input_x:np.random.randn(1, 3, 224, 224)})
        print('output_tf_pb = {}'.format(output_tf_pb))


