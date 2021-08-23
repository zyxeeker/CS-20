import tensorflow as tf

# tf1.X
tf.compat.v1.disable_eager_execution()

x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)
useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.compat.v1.Session() as sess:
    # run(fetches,feed_dict = None,options = None,run_metadata = None)
    # fetches(list)包含你想要得到的节点值
    z, not_useless = sess.run([pow_op, useless])

print(z, not_useless)

# 将图放到指定的硬件中
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
    c = tf.multiply(a, b)
# 查看硬件分配信息log
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
    sess.run(c)

# 创造一个图
g = tf.Graph()
# 在不生成图的情况下，添加op算子属于默认图
# 通过设置默认图，并在图中添加op算子才是用户所创造的图，如下
with g.as_default():
    x = tf.add(3, 5)
with tf.compat.v1.Session(graph=g) as sess:
    sess.run(x)

# 处理默认图
g = tf.compat.v1.get_default_graph()
