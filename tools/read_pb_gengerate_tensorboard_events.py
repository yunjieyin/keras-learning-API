import tensorflow as tf
import tensorboard

# model = ('D:\\work\\pb_file\\model-400000.pb')
# graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()
# graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
# tf.import_graph_def(graph_def, name='graph')
# summaryWriter = tf.summary.FileWriter('log/', graph)


# model = ('D:\Project\python\load_pb_file\pb_file\single_conv.pb')
# graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()
# graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
# tf.import_graph_def(graph_def, name='graph')
# summaryWriter = tf.summary.FileWriter('pb_file/', graph)


model = ('D:\Project\python\load_pb_file\pb_file\conv_act_pool\conv_act_pool.pb')
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(tf.gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('pb_file/conv_act_pool/', graph)





