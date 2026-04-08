import tensorflow as tf

B = tf.constant([
    [[700, 2, 53], [4, 5, 6], [7, 8, 91]],
    [[10, 11, 91], [13, 31, 15], [53, 77, 48]],
    [[700, 20, 21], [22, 100, 24], [25, 26, 27]]
])

max_val = tf.reduce_max(B)
indices = tf.where(tf.equal(B, max_val)) 
argmax = tf.argmax(B)
#tf.reduce_max(B)
#indices = tf.where(tf.equal(B, max_val))
tf.print(argmax)
#tf.print(indices)
