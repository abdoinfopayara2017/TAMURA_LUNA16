import tensorflow as tf
import numpy as np
from .utils import conv3d,argmax_tesor,calculate_theta,histogram_direction,maxlinelikeness_tf,split_sub_tensors


shared_dynamic_buffer_A = tf.Variable(
    initial_value=tf.zeros([1, 1, 1, 1, 1,1], dtype=tf.float32), 
    shape=tf.TensorShape([None, None, None, None, None, None]) # Allows resizing
)

shared_dynamic_buffer_E = tf.Variable(
    initial_value=tf.zeros([1, 1, 1, 1, 1, 1, 1], dtype=tf.float32), 
    shape=tf.TensorShape([None, None, None, None, None, None, None]) # Allows resizing
)

shared_dynamic_buffer_kernel = tf.Variable(
    initial_value=tf.zeros([1, 1, 1, 1, 1], dtype=tf.float32), 
    shape=tf.TensorShape([None, None, None, None, None]) # Allows resizing
)

shared_dynamic_buffer_tensor_temp = tf.Variable(
    initial_value=tf.zeros([1, 1, 1, 1, 1], dtype=tf.float32), 
    shape=tf.TensorShape([None, None, None, None, None]) # Allows resizing
)

shared_dynamic_buffer_kernal_z = tf.Variable(
    initial_value=tf.zeros([1, 1, 1, 1, 1], dtype=tf.float32), 
    shape=tf.TensorShape([None, None, None, None, None]) # Allows resizing
)

shared_dynamic_buffer_kernal_y = tf.Variable(
    initial_value=tf.zeros([1, 1, 1, 1, 1], dtype=tf.float32), 
    shape=tf.TensorShape([None, None, None, None, None]) # Allows resizing
)

shared_dynamic_buffer_kernal_x = tf.Variable(
    initial_value=tf.zeros([1, 1, 1, 1, 1], dtype=tf.float32), 
    shape=tf.TensorShape([None, None, None, None, None]) # Allows resizing
)

shared_dynamic_buffer_K_argmax = tf.Variable(
    initial_value=tf.zeros([1], dtype=tf.float32), 
    shape=tf.TensorShape([None]) # Allows resizing
)

shared_dynamic_buffer_K_argmax = tf.Variable(
    initial_value=tf.zeros([1], dtype=tf.float32), 
    shape=tf.TensorShape([None]) # Allows resizing
)

shared_dynamic_buffer_last_index = tf.Variable(
    initial_value=tf.zeros([1], dtype=tf.float32), 
    shape=tf.TensorShape([None]) # Allows resizing
)

shared_dynamic_buffer_indices = tf.Variable(
    initial_value=tf.zeros([1, 1], dtype=tf.float32), 
    shape=tf.TensorShape([None, None]) # Allows resizing
)


class TamuraMethod :
    
    def __init__(self,z , y , x):
        self.z=z
        self.y=y
        self.x=x
        self.image_with_batch = tf.Variable(tf.zeros([1,z,y,x,1],dtype=tf.float32))        
        self.A = shared_dynamic_buffer_A
        self.E = shared_dynamic_buffer_E        
        self.kernal = shared_dynamic_buffer_kernel
        self.tensor_temp = shared_dynamic_buffer_tensor_temp
        self.kernal_z = shared_dynamic_buffer_kernal_z
        self.kernal_y = shared_dynamic_buffer_kernal_y
        self.kernal_x = shared_dynamic_buffer_kernal_x
        self.Sbest = tf.Variable(tf.zeros([z,y,x],dtype=tf.float32))               
    
    
    # 3D Coarseness
    @tf.function
    def calc_coarseness(self,numpy_image , K_max ,t):
        '''
        numpy_image image in numpy format
        2^K_max denotes the maximum window size
        t is a certain constant less than 1
        
        '''
        #Retrieve image dimensions
        #z , y , x = numpy_image.shape
        #check K_max value
        K_max = K_max if (np.power(2,K_max) < self.z) else int(np.log(self.z) / np.log(2))
        K_max = K_max if (np.power(2,K_max) < self.y) else int(np.log(self.y) / np.log(2))
        K_max = K_max if (np.power(2,K_max) < self.x) else int(np.log(self.x) / np.log(2))
        
        # Convert to a TensorFlow tensor
        #numpy_image = tf.convert_to_tensor(numpy_image)
        
        self.image_with_batch[0,:,:,:,0].assign(numpy_image)
        #Average Grey Value Matrix A
        self.A.assign(tf.zeros([K_max,1,self.z,self.y,self.x,1],dtype=tf.float32))
        #Calculate the Ak difference between non-overlapping windows in 3 directions for each pixel point.    
        self.E.assign(tf.zeros([K_max,3,1,self.z,self.y,self.x,1],dtype=tf.float32))
        
        puissance = tf.Variable(0.0,dtype=tf.float32) #tf.cast(tf.math.pow(2,3*k),dtype=tf.float32)
        
        for k in range (K_max):
            puissance.assign(tf.cast(tf.math.pow(2,3*k),dtype=tf.float32))
            window = np.power(2,k)  
            self.kernal.assign(tf.ones(shape=[window,window,window,1,1]))
            self.tensor_temp.assign(conv3d(self.image_with_batch,self.kernal,1))
            self.tensor_temp.assign(tf.math.divide(self.tensor_temp,puissance))
            self.A[k,:,:,:,:,:].assign(self.tensor_temp)
            self.kernal_z.assign(tf.zeros((2*window+1,1,1,1,1)))
            self.kernal_z[0,0,0,0,0].assign(1.0)
            self.kernal_z[2*window,0,0,0,0].assign(-1.0)
            self.kernal_y.assign(tf.zeros((1,2*window+1,1,1,1)))
            self.kernal_y[0,0,0,0,0].assign(1.0)
            self.kernal_y[0,2*window,0,0,0].assign(-1.0)
            self.kernal_x.assign(tf.zeros((1,1,2*window+1,1,1)))
            self.kernal_x[0,0,0,0,0].assign(1.0)
            self.kernal_x[0,0,2*window,0,0].assign(-1.0)       
            self.E[k,0,:].assign(tf.math.abs(conv3d(self.A[k,:],self.kernal_z,1)))
            self.E[k,1,:].assign(tf.math.abs(conv3d(self.A[k,:],self.kernal_y,1)))
            self.E[k,2,:].assign(tf.math.abs(conv3d(self.A[k,:],self.kernal_x,1)))
        
        #For each pixel, compute the value of k that maximises E
        K_argmax = shared_dynamic_buffer_K_argmax
        last_index = shared_dynamic_buffer_last_index
        indices = shared_dynamic_buffer_indices
        E_max = tf.Variable(0.0,dtype=tf.float32)  
        self.Sbest.assign(tf.zeros([self.z,self.y,self.x],dtype=tf.float32))
        for iz in range (self.z):
            for iy in range (self.y):
                for ix in range (self.x):
                    K_argmax.assign(tf.cast(argmax_tesor(self.E[:,:,:,iz,iy,ix,:])[0],dtype=tf.float32))
                    E_max.assign(tf.cast(argmax_tesor(self.E[:,:,:,iz,iy,ix,:])[1],dtype=tf.float32))
                    number = tf.math.multiply(E_max,tf.constant(t))
                    indices.assign(tf.cast(tf.where(tf.greater(self.E[:,:,:,iz,iy,ix,:], number)),dtype=tf.float32))
                    K_argmax_pow = K_argmax[0]
                    if(tf.not_equal(tf.size(indices), 0)):
                        last_index.assign(tf.cast(tf.reduce_max(indices,0),dtype=tf.float32))
                        #last_index.assign(last_index[0],validate_shape=False)
                        #K_argmax.assign(K_argmax[0],validate_shape=False)
                        K_argmax_pow = tf.where(tf.greater(K_argmax[0],last_index[0]),K_argmax[0],last_index[0])
                    self.Sbest[iz,iy,ix].assign(tf.cast(tf.math.pow(tf.constant(2),tf.cast(K_argmax_pow,tf.int32)),dtype=tf.float32))
                        
        Fcrs = tf.reduce_mean(self.Sbest)
        
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
        

        
          





