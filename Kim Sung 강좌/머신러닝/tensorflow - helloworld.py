import tensorflow as tf


# default graph 형성
hello = tf.constant("Hello, TensorFlow!")

# session을 형성
sess = tf.Session()

# session 실행
print(sess.run(hello))

# b -> byte string인것 
