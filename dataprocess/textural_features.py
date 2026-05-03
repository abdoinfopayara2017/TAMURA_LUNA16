import numpy as np
from scipy.ndimage import convolve
from .utils import maxlinelikeness,standard_deviation

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
    
    #Average Grey Value Matrix A
    A = np.zeros([K_max,z,y,x])
    #Calculate the Ak difference between non-overlapping windows in 3 directions for each pixel point.    
    E = np.zeros([K_max,3,z,y,x]) 
    
    for k in range (K_max):
     window = np.power(2,k)    
     for iz in range (z)[window : z-window]:
        for iy in range (y)[window : y-window]:
            for ix in range (x)[window : x-window]:
               A[k,iz,iy,ix] = np.sum(numpy_image[iz-window:iz+window,iy-window:iy+window,ix-window:ix+window])
               A[k,iz,iy,ix] = A[k,iz,iy,ix] / np.power(2,3*k)
     for iz in range (z)[window : z-window]:
        for iy in range (y)[window : y-window]:
           for ix in range (x)[window : x-window]:
               E[k,0,iz,iy,ix] = np.abs(A[k,iz+window,iy,ix] - A[k,iz-window,iy,ix])
               E[k,1,iz,iy,ix] = np.abs(A[k,iz,iy+window,ix] - A[k,iz,iy-window,ix])
               E[k,2,iz,iy,ix] = np.abs(A[k,iz,iy,ix+window] - A[k,iz,iy,ix-window])
        
    
    #For each pixel, compute the value of k that maximises E
    Sbest = np.zeros([z,y,x])
    for iz in range (z):
        for iy in range (y):
            for ix in range (x):
                #At each point, pick the best size which gives the highest output value
                flat_index = np.argmax(E[:,:,iz,iy,ix])
                # Convert the flat index to a tuple of coordinates                
                K_argmax,direction = np.unravel_index(flat_index, E[:,:,iz,iy,ix].shape)
                E_max = E[K_argmax,direction,iz,iy,ix]               
                number = t * E_max
                indices = np.where(E[:,:,iz,iy,ix] >= number)
                #Get the last index
                if len(indices[0]) and number :
                    last_index = np.max(indices[0])
                    #if there exist some k such that k > K_argmax which gives  E_max and Ek >= tE.., we take the largest k for S best
                    K_argmax = K_argmax if (K_argmax > last_index) else last_index
                Sbest[iz,iy,ix] = np.power(2,K_argmax)
         
    #The mean of all Sbest values serves as the Coarseness of the entire image.                       
    F_crs = np.mean(Sbest) 
    return F_crs               

def contrast(numpy_image):
    numpy_image = np.reshape(numpy_image, (1, numpy_image.shape[0]*numpy_image.shape[1]*numpy_image.shape[2]))
    
    m4 = np.mean(np.power(numpy_image - np.mean(numpy_image),4))
    v = np.var(numpy_image)
    std = np.power(v, 0.5)    
    if v :
     alfa4 = m4 / np.power(v,2)
     fcon = std / np.power(alfa4, 0.25)
     return fcon
    else : 
     return 0                
    
def directionality(numpy_image):
    #Retrieve image dimensions
    z , y , x = numpy_image.shape
    
    # This kernel highlights changes along the z-axis (depth).
    z_direction = np.array([
    [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]], 
    [[ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0]], 
    [[ 1,  1,  1], [ 1,  1,  1], [ 1,  1,  1]]  
                         ])
    
    # This kernel highlights changes along the y-axis (top-to-bottom).
    y_direction = np.array([
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], 
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], 
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]  
                           ])
    
    # This kernel highlights changes along the x-axis (left-to-right).
    x_direction = np.array([
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], 
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], 
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]  
                          ])
    
    deltaZ = np.zeros([z,y,x])
    deltaY = np.zeros([z,y,x])
    deltaX = np.zeros([z,y,x])
    
    #local edge directions theta
    thetaYX =  np.zeros([z,y,x])
    thetaZY =  np.zeros([z,y,x])
    thetaXZ =  np.zeros([z,y,x])
    
    #the magnitude deltaG
    deltaGYX =  np.zeros([z,y,x])
    deltaGZY =  np.zeros([z,y,x])
    deltaGXZ =  np.zeros([z,y,x])
	
    # calc for deltaZ
    deltaZ =  convolve(numpy_image, z_direction, mode='constant', cval=0.0)
    # calc for deltaY
    deltaY =  convolve(numpy_image, y_direction, mode='constant', cval=0.0)   
    # calc for deltaX
    deltaX =  convolve(numpy_image, x_direction, mode='constant', cval=0.0)  
    
    deltaGYX = (np.absolute(deltaX) + np.absolute(deltaY)) / 2.0
    deltaGYX_dir = deltaGYX
    deltaGYX = np.reshape(deltaGYX, (deltaGYX.shape[0] * deltaGYX.shape[1] * deltaGYX.shape[2]))
    
    deltaGZY = (np.absolute(deltaZ) + np.absolute(deltaY)) / 2.0
    deltaGZY_dir = deltaGZY
    deltaGZY = np.reshape(deltaGZY, (deltaGZY.shape[0] * deltaGZY.shape[1] * deltaGZY.shape[2]))
    
    deltaGXZ = (np.absolute(deltaZ) + np.absolute(deltaX)) / 2.0
    deltaGXZ_dir = deltaGXZ
    deltaGXZ = np.reshape(deltaGXZ, (deltaGXZ.shape[0] * deltaGXZ.shape[1] * deltaGXZ.shape[1]))  
    
    # calc the thetaYX
    for yi in range(y):
        for xi in range(x):
            for zi in range(z):
                if (deltaY[yi][xi][zi] == 0 and deltaX[yi][xi][zi] == 0):
                    thetaYX[yi][xi][zi] = 0
                elif(deltaX[yi][xi][zi] == 0):
                    thetaYX[yi][xi][zi] = np.pi
                else:
                    thetaYX[yi][xi][zi] = np.arctan(deltaY[yi][xi][zi] / deltaX[yi][xi][zi]) + np.pi / 2.0
    thetaYX_dir = thetaYX
    thetaYX = np.reshape(thetaYX, (thetaYX.shape[0] * thetaYX.shape[1] * thetaYX.shape[2]))
    
    # calc the thetaZY
    for yi in range(y):
        for xi in range(x):
            for zi in range(z):
                if (deltaZ[yi][xi][zi] == 0 and deltaY[yi][xi][zi] == 0):
                    thetaZY[yi][xi][zi] = 0
                elif(deltaY[yi][xi][zi] == 0):
                    thetaZY[yi][xi][zi] = np.pi
                else:
                    thetaZY[yi][xi][zi] = np.arctan(deltaZ[yi][xi][zi] / deltaY[yi][xi][zi]) + np.pi / 2.0
    thetaZY_dir = thetaZY 
    thetaZY = np.reshape(thetaZY, (thetaZY.shape[0] * thetaZY.shape[1] * thetaZY.shape[2]))
    
    # calc the thetaXZ
    for yi in range(y):
        for xi in range(x):
            for zi in range(z):
                if (deltaX[yi][xi][zi] == 0 and deltaZ[yi][xi][zi] == 0):
                    thetaXZ[yi][xi][zi] = 0
                elif(deltaZ[yi][xi][zi] == 0):
                    thetaXZ[yi][xi][zi] = np.pi
                else:
                    thetaXZ[yi][xi][zi] = np.arctan(deltaX[yi][xi][zi] / deltaZ[yi][xi][zi]) + np.pi / 2.0
    thetaXZ_dir = thetaXZ
    thetaXZ = np.reshape(thetaXZ, (thetaXZ.shape[0] * thetaXZ.shape[1] *  thetaXZ.shape[2]))
    
    n = 16
    t = 12
    hd_thetaYX =  np.zeros(n)
    hd_thetaZY =  np.zeros(n)
    hd_thetaXZ =  np.zeros(n)
    
    # calc the direction with respect to thetaXZ
    dlen = deltaGXZ.shape[0]
    for ni in range(n):
        for k in range(dlen):
            if((deltaGXZ[k] >= t) and (thetaXZ[k] >= (2*ni-1) * np.pi / (2 * n)) and (thetaXZ[k] < (2*ni+1) * np.pi / (2 * n))):
                hd_thetaXZ[ni] += 1
    hd_thetaXZ = hd_thetaXZ / np.sum(hd_thetaXZ)
    hd_thetaXZ_max_index = np.argmax(hd_thetaXZ)
    
    # calc the direction with respect to thetaYX
    dlen = deltaGYX.shape[0]
    for ni in range(n):
        for k in range(dlen):
            if((deltaGYX[k] >= t) and (thetaYX[k] >= (2*ni-1) * np.pi / (2 * n)) and (thetaYX[k] < (2*ni+1) * np.pi / (2 * n))):
                hd_thetaYX[ni] += 1
    hd_thetaYX = hd_thetaYX / np.sum(hd_thetaYX)
    hd_thetaYX_max_index = np.argmax(hd_thetaYX)
    
    # calc the direction with respect to thetaZY
    dlen = deltaGZY.shape[0]
    for ni in range(n):
        for k in range(dlen):
            if((deltaGZY[k] >= t) and (thetaZY[k] >= (2*ni-1) * np.pi / (2 * n)) and (thetaZY[k] < (2*ni+1) * np.pi / (2 * n))):
                hd_thetaZY[ni] += 1
    hd_thetaZY = hd_thetaZY / np.sum(hd_thetaZY)
    hd_thetaZY_max_index = np.argmax(hd_thetaZY)
    
    F_dir = 0
    F_dirXZ = 0
    F_dirYX = 0
    F_dirZY = 0
    for ni in range(n):
        F_dirXZ += np.power((ni - hd_thetaXZ_max_index), 2) * hd_thetaXZ[ni] 
        F_dirYX += (np.power((ni - hd_thetaYX_max_index), 2) * hd_thetaYX[ni]) 
        F_dirZY += (np.power((ni - hd_thetaZY_max_index), 2) * hd_thetaZY[ni])
    
    F_dir = F_dirXZ if(F_dirXZ > F_dirYX) else F_dirYX
    F_dir = F_dir if(F_dir > F_dirZY) else F_dirZY
    
    return F_dir , thetaXZ_dir , deltaGXZ_dir, thetaYX_dir , deltaGYX_dir, thetaZY_dir, deltaGZY_dir 

def linelikeness(numpy_image):
    _,theta_1,delta_1,theta_2,delta_2,theta_3,delta_3 =  directionality(numpy_image)
    
   
    F_linelikeness = np.zeros(3)
    F_linelikeness[0] = maxlinelikeness(theta_1,delta_1,4,12,16)
    F_linelikeness[1] = maxlinelikeness(theta_2,delta_2,4,12,16)
    F_linelikeness[2] = maxlinelikeness(theta_3,delta_3,4,12,16)
    
    F_lin = np.max(F_linelikeness)
    return F_lin

def regularity(numpy_image):
    shape = np.array(numpy_image.shape)   
    windows_size = np.prod(shape) / 8
    windows_size = np.round(np.power(windows_size,1/3)).astype(int)
    
    CRS = np.zeros(8)
    COS = np.zeros(8)
    DIR = np.zeros(8)
    LIN = np.zeros(8)
    k = 0
    for zi in range (0,shape[0],windows_size):
       for yi in range (0,shape[1],windows_size):
         for xi in range (0,shape[2],windows_size):
             CRS[k] = calc_coarseness(numpy_image[zi:zi+windows_size,yi:yi+windows_size,xi:xi+windows_size],5,0.9)
             COS[k] = contrast(numpy_image[zi:zi+windows_size,yi:yi+windows_size,xi:xi+windows_size])
             DIR[k],*rest = directionality(numpy_image[zi:zi+windows_size,yi:yi+windows_size,xi:xi+windows_size])
             LIN[k] = linelikeness(numpy_image[zi:zi+windows_size,yi:yi+windows_size,xi:xi+windows_size])
             k += 1
    
    
    # calculat standard deviation
    F_reg = standard_deviation(COS) + standard_deviation(CRS) + standard_deviation(DIR) + standard_deviation(LIN) 
    
    return F_reg

def roughness(numpy_image):
    F_rgh = calc_coarseness(numpy_image,5,0.9) + contrast(numpy_image)
    return F_rgh        
            
       
    
   
        
    
	
                 

                        
                
    
