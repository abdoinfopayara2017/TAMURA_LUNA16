import tensorflow as tf
import numpy as np
from .utils import conv3d,argmax_tesor,calculate_theta,histogram_direction,maxlinelikeness_tf,split_sub_tensors

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

def contrast(numpy_image):
    
    # Convert to a TensorFlow tensor
    numpy_image = tf.convert_to_tensor(numpy_image)
    # Reshape to 1D
    numpy_image = tf.reshape(numpy_image, [-1]) 
    m4 = tf.reduce_mean(tf.math.pow(tf.subtract(numpy_image,tf.reduce_mean(numpy_image)),4))
    v = tf.math.reduce_variance(numpy_image)
    std = tf.pow(v, 0.5)
    if(tf.not_equal(v,tf.constant(0,dtype=tf.float32))):
        alfa4 = m4 / tf.pow(v,2)
        fcon = std / tf.pow(alfa4, 0.25)
        return tf.reshape(fcon,[-1])
    else : 
     return tf.reshape(tf.constant(0),[-1])

def directionality(numpy_image):
    #Retrieve image dimensions
    z , y , x = numpy_image.shape
    # This kernel highlights changes along the z-axis (depth).
    z_direction = tf.constant([
    [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]], 
    [[ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0]], 
    [[ 1,  1,  1], [ 1,  1,  1], [ 1,  1,  1]]  
                         ],dtype=tf.float32)
    z_kernal = tf.Variable(tf.zeros([3,3,3,1,1],dtype=tf.float32))
    z_kernal[:,:,:,0,0].assign(z_direction)
    
    # This kernel highlights changes along the y-axis (top-to-bottom).
    y_direction = tf.constant([
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], 
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], 
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]  
                           ],dtype=tf.float32)
    
    y_kernal = tf.Variable(tf.zeros([3,3,3,1,1],dtype=tf.float32))
    y_kernal[:,:,:,0,0].assign(y_direction)
    
    # This kernel highlights changes along the x-axis (left-to-right).
    x_direction = tf.constant([
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], 
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], 
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]  
                          ],dtype=tf.float32)
    
    x_kernal = tf.Variable(tf.zeros([3,3,3,1,1],dtype=tf.float32))
    x_kernal[:,:,:,0,0].assign(x_direction)

    deltaZ = tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))
    deltaY = tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))
    deltaX = tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))

    #local edge directions theta
    thetaYX =  tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))
    thetaZY =  tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))
    thetaXZ =  tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))  

    #the magnitude deltaG
    deltaGYX =  tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))
    deltaGZY =  tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))
    deltaGXZ =  tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))
     
    #convert and reshape
    numpy_image = tf.convert_to_tensor(numpy_image)
    image_with_batch = tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))
    image_with_batch[0,:,:,:,0].assign(numpy_image)
    
    
    # calc for deltaZ
    deltaZ =  conv3d(image_with_batch, z_kernal,1)
    # calc for deltaY
    deltaY =  conv3d(image_with_batch, y_kernal,1)   
    # calc for deltaX
    deltaX =  conv3d(image_with_batch, x_kernal,1)  

    deltaGYX = (tf.abs(deltaX) + tf.abs(deltaY)) / 2.0
    deltaGYX_dir = deltaGYX
    deltaGYX = tf.reshape(deltaGYX, [-1])
    
    deltaGZY = (tf.abs(deltaZ) + tf.abs(deltaY)) / 2.0
    deltaGZY_dir = deltaGZY
    deltaGZY = tf.reshape(deltaGZY, [-1])
    
    deltaGXZ = (tf.abs(deltaZ) + tf.abs(deltaX)) / 2.0
    deltaGXZ_dir = deltaGXZ
    deltaGXZ = tf.reshape(deltaGXZ, [-1])

    #calculate thetas
    thetaYX.assign(calculate_theta(deltaY,deltaX))
    thetaYX_dir = thetaYX
    thetaYX = tf.reshape(thetaYX,[-1])
    
    thetaZY.assign(calculate_theta(deltaZ,deltaY))
    thetaZY_dir = thetaZY
    thetaZY = tf.reshape(thetaZY,[-1])

    thetaXZ.assign(calculate_theta(deltaX,deltaZ))
    thetaXZ_dir = thetaXZ
    thetaXZ = tf.reshape(thetaXZ,[-1])

    # calc the direction with respect to thetaXZ
    F_dirXZ = histogram_direction(thetaXZ , deltaGXZ)
    F_dirYX = histogram_direction(thetaYX , deltaGYX)
    F_dirZY = histogram_direction(thetaZY , deltaGZY)

    F_dir = F_dirXZ if(tf.greater(F_dirXZ,F_dirYX)) else F_dirYX
    F_dir = F_dir if(tf.greater(F_dir,F_dirZY)) else F_dirZY

    return tf.reshape(F_dir,[-1]) , thetaXZ_dir , deltaGXZ_dir, thetaYX_dir , deltaGYX_dir, thetaZY_dir, deltaGZY_dir 

def linelikeness(numpy_image):
    _,theta_1,delta_1,theta_2,delta_2,theta_3,delta_3 =  directionality(numpy_image)
    
   
    F_linelikeness = tf.Variable(tf.zeros([3],dtype=tf.float32))
    F_linelikeness = tf.tensor_scatter_nd_update(F_linelikeness,[[0]],maxlinelikeness_tf(theta_1,delta_1,4,12,16))
    
    F_linelikeness = tf.tensor_scatter_nd_update(F_linelikeness,[[1]],maxlinelikeness_tf(theta_2,delta_2,4,12,16))
    
    F_linelikeness = tf.tensor_scatter_nd_update(F_linelikeness,[[2]],maxlinelikeness_tf(theta_3,delta_3,4,12,16))
        
    F_lin = tf.reduce_max(F_linelikeness)
    
    F_lin = tf.expand_dims(F_lin, axis=0)
    return F_lin

def regularity(numpy_image):
    shape = numpy_image.shape   
    windows_size = tf.reduce_prod(shape) / 8
    windows_size = tf.round(tf.pow(windows_size,1/3))
    windows_size = tf.cast(windows_size,tf.int32)
    
    CRS = tf.zeros(0,dtype=tf.float32)
    COS = tf.zeros(0,dtype=tf.float32)
    DIR = tf.zeros(0,dtype=tf.float32)
    LIN = tf.zeros(0,dtype=tf.float32)
    sub_tensors = split_sub_tensors(numpy_image)
    for s in range (8):        
        CRS = tf.concat([CRS,calc_coarseness(sub_tensors[s],5,0.9)],0)
        COS = tf.concat([COS,contrast(sub_tensors[s])],0)
        DIR = tf.concat([DIR,directionality(sub_tensors[s])[0]],0)
        LIN = tf.concat([LIN,linelikeness(sub_tensors[s])],0)
    
    # calculat standard deviation
    F_reg = tf.add(tf.math.reduce_std(CRS),tf.math.reduce_std(COS))
    F_reg = tf.add(F_reg,tf.math.reduce_std(DIR))
    F_reg = tf.add(F_reg,tf.math.reduce_std(LIN))
    F_reg = tf.expand_dims(F_reg, axis=0)
    return F_reg

def roughness(F_crs,F_cos):
    F_rgh = tf.add(F_crs,F_cos)
    return F_rgh 
        

        
          





