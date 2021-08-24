import tensorflow as tf
import os

# 降低输出log等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# tf1.X
tf.compat.v1.disable_eager_execution()

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

writer = tf.compat.v1.summary.FileWriter('./graphs', tf.compat.v1.get_default_graph())
with tf.compat.v1.Session() as sess:
    print(sess.run(x))
writer.close()

# Constant
# tf.constant(value,dtype=None,shape=None,name='Const',verify_shape=False)
a = tf.constant([2, 2], name='a')
b = tf.constant([[2, 2], [3, 2]], name='b')
x = tf.multiply(a, b, name='mul')

with tf.compat.v1.Session() as sess:
    print(sess.run(x))

# 0填充张量
# tf.zeros(shape, dtype=tf.float32, name=None)
# 将输入的张量中的所有元素全部置为0
# tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
# 张量每行大小要一样
input_tensor = tf.constant([[1, 2], [2, 3], [4, 6]])
a = tf.zeros([2, 3], tf.int32)
c = tf.zeros_like(input_tensor)

with tf.compat.v1.Session() as sess:
    print("将元素置为0")
    print(sess.run(c))

# 将其中的元素置为1,意义同上
# tf.ones(shape, dtype=tf.float32, name=None)
# tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
a = tf.ones([2, 3], dtype=tf.int32)
b = tf.ones_like(input_tensor)
with tf.compat.v1.Session() as sess:
    print("将元素置为1")
    print(sess.run(a))
    print(sess.run(b))

# 用指定值填充新建张量
# tf.fill(dims, value, name=None)
a = tf.fill([2, 3], 8)
with tf.compat.v1.Session() as sess:
    print("填充指定值")
    print(sess.run(a))
