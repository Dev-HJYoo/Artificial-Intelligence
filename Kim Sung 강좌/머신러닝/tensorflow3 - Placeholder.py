import tensorflow as tf


# 나중에 입력받겠다는 것
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

# 세션 형
sess = tf.Session()

# 여기서 입력을 주는 것
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2, 4]}))
