import tensorflow as tf

rows = 5
cols = 5
# Create meshgrid based on row/column counts
r, _ = tf.meshgrid(tf.range(rows),tf.range(cols), indexing='ij')
print(r)