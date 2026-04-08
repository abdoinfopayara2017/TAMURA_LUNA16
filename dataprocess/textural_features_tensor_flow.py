import tensorflow as tf
import numpy as np
from .utils import conv3d,argmax_tesor

# 3D Coarseness
def calc_coarseness(numpy_image , K_max ,t):
    '''
    numpy_image image in numpy format
    2^K_max denotes the maximum window size
    t is a certain constant less than 1
    
    '''
    #Retrieve image dimensions
    z , y , x = numpy_image.shape
    #check K_max value
    K_max = K_max if (np.power(2,K_max) < z) else int(np.log(z) / np.log(2))
    K_max = K_max if (np.power(2,K_max) < y) else int(np.log(y) / np.log(2))
    K_max = K_max if (np.power(2,K_max) < x) else int(np.log(x) / np.log(2))
    
    # Convert to a TensorFlow tensor
    numpy_image = tf.convert_to_tensor(numpy_image)
    image_with_batch = tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))
    image_with_batch[0,:,:,:,0].assign(numpy_image)
    #Average Grey Value Matrix A
    A = tf.Variable(tf.zeros([K_max,1,z,y,x,1],dtype=tf.float32))
    #Calculate the Ak difference between non-overlapping windows in 3 directions for each pixel point.    
    E = tf.Variable(tf.zeros([K_max,3,1,z,y,x,1],dtype=tf.float32))

    for k in range (K_max):
        window = np.power(2,k)  
        kernal = tf.ones(shape=[window,window,window,1,1])
        tensor_temp = conv3d(image_with_batch,kernal,1)
        tensor_temp = tf.math.divide(tensor_temp,tf.cast(tf.math.pow(2,3*k),dtype=tf.float32))
        A[k,:,:,:,:,:].assign(tensor_temp)
        kernal_z = np.zeros((2*window+1,1,1,1,1))
        kernal_z[0,0,0,0,0] = 1
        kernal_z[2*window,0,0,0,0] = -1
        kernal_y = np.zeros((1,2*window+1,1,1,1))
        kernal_y[0,0,0,0,0] = 1
        kernal_y[0,2*window,0,0,0] = -1
        kernal_x = np.zeros((1,1,2*window+1,1,1))
        kernal_x[0,0,0,0,0] = 1
        kernal_x[0,0,2*window,0,0] = -1       
        E[k,0,:].assign(tf.math.abs(conv3d(A[k,:],tf.convert_to_tensor(kernal_z,dtype=tf.float32),1)))
        E[k,1,:].assign(tf.math.abs(conv3d(A[k,:],tf.convert_to_tensor(kernal_y,dtype=tf.float32),1)))
        E[k,2,:].assign(tf.math.abs(conv3d(A[k,:],tf.convert_to_tensor(kernal_x,dtype=tf.float32),1)))
    
    #For each pixel, compute the value of k that maximises E
    Sbest = tf.Variable(tf.zeros([z,y,x],dtype=tf.float32))
    for iz in range (z):
        for iy in range (y):
            for ix in range (x):
                K_argmax,E_max = argmax_tesor(E[:,:,:,iz,iy,ix,:])
                number = tf.math.multiply(E_max,tf.constant(t))
                indices = tf.where(tf.greater(E[:,:,:,iz,iy,ix,:], number))
                if(tf.not_equal(tf.size(indices), 0)):
                    last_index = tf.reduce_max(indices,0)
                    last_index = last_index[0]
                    K_argmax = K_argmax[0]
                    K_argmax = K_argmax if (tf.greater(K_argmax,last_index)) else last_index
                Sbest[iz,iy,ix].assign(tf.cast(tf.math.pow(tf.constant(2),tf.cast(K_argmax,tf.int32)),dtype=tf.float32))
                    
    Fcrs = tf.reduce_mean(Sbest)
    
    return  tf.reshape(Fcrs, [1])       
    


        

        
          





