
import numpy as np
import tensorflow as tf

def maxlinelikeness(theta,delta,distance,threshold,size):
    
    z , y , x = theta.shape
    n = size
    # 2 mens -1
    #Axial (3)
    Pd_axial_1_0_0 = np.zeros([n,n])
    Pd_axial_0_1_0 = np.zeros([n,n])
    Pd_axial_0_0_1 = np.zeros([n,n])    
    
    #symmetric    
    Pd_axial_2_0_0 = np.zeros([n,n])
    Pd_axial_0_2_0 = np.zeros([n,n])
    Pd_axial_0_0_2 = np.zeros([n,n]) 
    
    #Planar Diagonals (6)     
    Pd_planar_diagonal_1_1_0 = np.zeros([n,n])
    Pd_planar_diagonal_1_2_0 = np.zeros([n,n])
    Pd_planar_diagonal_1_0_1 = np.zeros([n,n])
    Pd_planar_diagonal_1_0_2 = np.zeros([n,n])
    Pd_planar_diagonal_0_1_1 = np.zeros([n,n])
    Pd_planar_diagonal_0_1_2 = np.zeros([n,n])
    
    #symmetric
    Pd_planar_diagonal_2_2_0 = np.zeros([n,n])
    Pd_planar_diagonal_2_1_0 = np.zeros([n,n])
    Pd_planar_diagonal_2_0_2 = np.zeros([n,n])
    Pd_planar_diagonal_2_0_1 = np.zeros([n,n])
    Pd_planar_diagonal_0_2_2 = np.zeros([n,n])
    Pd_planar_diagonal_0_2_1 = np.zeros([n,n])
    
    #Volumetric Diagonals (4)
    Pd_volumetric_diagonal_1_1_1 = np.zeros([n,n])
    Pd_volumetric_diagonal_1_1_2 = np.zeros([n,n])
    Pd_volumetric_diagonal_1_2_1 = np.zeros([n,n])
    Pd_volumetric_diagonal_1_2_2 = np.zeros([n,n])
    
    #symmetric
    Pd_volumetric_diagonal_2_2_2 = np.zeros([n,n])
    Pd_volumetric_diagonal_2_2_1 = np.zeros([n,n])
    Pd_volumetric_diagonal_2_1_2 = np.zeros([n,n])
    Pd_volumetric_diagonal_2_1_1 = np.zeros([n,n])
    
    for zi in range (distance , z - distance):
        for yi in range (distance , y - distance):
            for xi in range (distance , x - distance):
                for m1 in range (n):
                    for m2 in range(n):
                        #Pd_axial_1_0_0
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi+distance,yi,xi]>=(2*(m2-1)*np.pi/2/n) and theta[zi+distance,yi,xi]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi+distance,yi,xi] >= threshold)) :
                            Pd_axial_1_0_0[m1,m2] += 1
                           
                            
                        #Pd_axial_2_0_0
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi-distance,yi,xi]>=(2*(m2-1)*np.pi/2/n) and theta[zi-distance,yi,xi]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi-distance,yi,xi] >= threshold)) :
                            Pd_axial_2_0_0[m1,m2] += 1
                            
                        #Pd_axial_0_1_0
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi,yi+distance,xi]>=(2*(m2-1)*np.pi/2/n) and theta[zi,yi+distance,xi]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi,yi+distance,xi] >= threshold)) :
                            Pd_axial_0_1_0[m1,m2] += 1
                        #Pd_axial_0_2_0
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi,yi-distance,xi]>=(2*(m2-1)*np.pi/2/n) and theta[zi,yi-distance,xi]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi,yi-distance,xi] >= threshold)) :
                            Pd_axial_0_2_0[m1,m2] += 1
                        
                        #Pd_axial_0_0_1
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi,yi,xi+distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi,yi,xi+distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi,yi,xi+distance] >= threshold)) :
                            Pd_axial_0_0_1[m1,m2] += 1
                        #Pd_axial_0_0_2
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi,yi,xi-distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi,yi,xi-distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi,yi,xi-distance] >= threshold)) :
                            Pd_axial_0_0_2[m1,m2] += 1
                            
                        
                        
                        #Pd_planar_diagonal_1_1_0
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi+distance,yi+distance,xi]>=(2*(m2-1)*np.pi/2/n) and theta[zi+distance,yi+distance,xi]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi+distance,yi+distance,xi] >= threshold)) :
                            Pd_planar_diagonal_1_1_0[m1,m2] += 1
                        #Pd_planar_diagonal_2_2_0
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi-distance,yi-distance,xi]>=(2*(m2-1)*np.pi/2/n) and theta[zi-distance,yi-distance,xi]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi-distance,yi-distance,xi])) :
                            Pd_planar_diagonal_2_2_0[m1,m2] += 1
                        
                        #Pd_planar_diagonal_1_2_0
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi+distance,yi-distance,xi]>=(2*(m2-1)*np.pi/2/n) and theta[zi+distance,yi-distance,xi]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi+distance,yi-distance,xi] >= threshold)) :
                            Pd_planar_diagonal_1_2_0[m1,m2] += 1
                        #Pd_planar_diagonal_2_1_0
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi-distance,yi+distance,xi]>=(2*(m2-1)*np.pi/2/n) and theta[zi-distance,yi+distance,xi]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi-distance,yi+distance,xi] >= threshold)) :
                            Pd_planar_diagonal_2_1_0[m1,m2] += 1
                        
                        #Pd_planar_diagonal_1_0_1
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi+distance,yi,xi+distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi+distance,yi,xi+distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi+distance,yi,xi+distance] >= threshold)) :
                            Pd_planar_diagonal_1_0_1[m1,m2] += 1
                        #Pd_planar_diagonal_2_0_2
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi-distance,yi,xi-distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi-distance,yi,xi-distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi-distance,yi,xi-distance] >= threshold)) :
                            Pd_planar_diagonal_2_0_2[m1,m2] += 1
                        
                        #Pd_planar_diagonal_1_0_2
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi+distance,yi,xi-distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi+distance,yi,xi-distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi+distance,yi,xi-distance]  >= threshold)) :
                            Pd_planar_diagonal_1_0_2[m1,m2] += 1
                        #Pd_planar_diagonal_2_0_1
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi-distance,yi,xi+distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi-distance,yi,xi+distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi-distance,yi,xi+distance] >= threshold)) :
                            Pd_planar_diagonal_2_0_1[m1,m2] += 1
                        
                        #Pd_planar_diagonal_0_1_1
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi,yi+distance,xi+distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi,yi+distance,xi+distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi,yi+distance,xi+distance] >= threshold)) :
                            Pd_planar_diagonal_0_1_1[m1,m2] += 1
                        #Pd_planar_diagonal_0_2_2
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi,yi-distance,xi-distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi,yi-distance,xi-distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi,yi-distance,xi-distance] >= threshold)) :
                            Pd_planar_diagonal_0_2_2[m1,m2] += 1
                        
                        #Pd_planar_diagonal_0_1_2
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi,yi+distance,xi-distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi,yi+distance,xi-distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi,yi+distance,xi-distance] >= threshold)) :
                            Pd_planar_diagonal_0_1_2[m1,m2] += 1
                        #Pd_planar_diagonal_0_2_1
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi,yi-distance,xi+distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi,yi-distance,xi+distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi,yi-distance,xi+distance] >= threshold)) :
                            Pd_planar_diagonal_0_2_1[m1,m2] += 1
                            
                        #Pd_volumetric_diagonal_1_1_1
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi+distance,yi+distance,xi+distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi+distance,yi+distance,xi+distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi+distance,yi+distance,xi+distance])) :
                            Pd_volumetric_diagonal_1_1_1[m1,m2] += 1
                        #Pd_volumetric_diagonal_2_2_2
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi-distance,yi-distance,xi-distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi-distance,yi-distance,xi-distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi-distance,yi-distance,xi-distance] >= threshold)) :
                            Pd_volumetric_diagonal_2_2_2[m1,m2] += 1
                            
                        #Pd_volumetric_diagonal_1_1_2
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi+distance,yi+distance,xi-distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi+distance,yi+distance,xi-distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi+distance,yi+distance,xi-distance])) :
                            Pd_volumetric_diagonal_1_1_2[m1,m2] += 1
                        #Pd_volumetric_diagonal_2_2_1
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi-distance,yi-distance,xi+distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi-distance,yi-distance,xi+distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi-distance,yi-distance,xi+distance])) :
                            Pd_volumetric_diagonal_2_2_1[m1,m2] += 1
                            
                        #Pd_volumetric_diagonal_1_2_1
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi+distance,yi-distance,xi+distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi+distance,yi-distance,xi+distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi+distance,yi-distance,xi+distance] >= threshold)) :
                            Pd_volumetric_diagonal_1_2_1[m1,m2] += 1
                        #Pd_volumetric_diagonal_2_1_2
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi-distance,yi+distance,xi-distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi-distance,yi+distance,xi-distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi-distance,yi+distance,xi-distance] >= threshold)) :
                            Pd_volumetric_diagonal_2_1_2[m1,m2] += 1
                            
                        #Pd_volumetric_diagonal_1_2_2
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi+distance,yi-distance,xi-distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi+distance,yi-distance,xi-distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi+distance,yi-distance,xi-distance] >= threshold)) :
                            Pd_volumetric_diagonal_1_2_2[m1,m2] += 1
                        #Pd_volumetric_diagonal_2_1_1
                        if ((theta[zi,yi,xi]>=(2*(m1-1)*np.pi/2/n) and theta[zi,yi,xi]<((2*(m1-1)+1)*np.pi/2/n)) and
                           (theta[zi-distance,yi+distance,xi+distance]>=(2*(m2-1)*np.pi/2/n) and theta[zi-distance,yi+distance,xi+distance]<((2*(m2-1)+1)*np.pi/2/n)) and 
                           (delta[zi,yi,xi] >= threshold and delta[zi-distance,yi+distance,xi+distance] >= threshold)) :
                            Pd_volumetric_diagonal_2_1_1[m1,m2] += 1       
    
    
    
    f = np.zeros(26)   
    g = np.zeros(26)
    
    for i in range(n):
        for j in range(n):
            f[0]=f[0]+Pd_axial_1_0_0[i,j]*np.cos((i-j)*2*np.pi/n)
            g[0]=g[0]+Pd_axial_1_0_0[i,j]            
            
            f[1]=f[1]+Pd_axial_2_0_0[i,j]*np.cos((i-j)*2*np.pi/n)
            g[1]=g[1]+Pd_axial_2_0_0[i,j]
            
            f[2]=f[2]+Pd_axial_0_1_0[i,j]*np.cos((i-j)*2*np.pi/n)
            g[2]=g[2]+Pd_axial_0_1_0[i,j]            
            f[3]=f[3]+Pd_axial_0_2_0[i,j]*np.cos((i-j)*2*np.pi/n)
            g[3]=g[3]+Pd_axial_0_2_0[i,j]
            
            f[4]=f[4]+Pd_axial_0_0_1[i,j]*np.cos((i-j)*2*np.pi/n)
            g[4]=g[4]+Pd_axial_0_0_1[i,j]            
            f[5]=f[5]+Pd_axial_0_0_2[i,j]*np.cos((i-j)*2*np.pi/n)
            g[5]=g[5]+Pd_axial_0_0_2[i,j]
            
            f[6]=f[6]+Pd_planar_diagonal_1_1_0[i,j]*np.cos((i-j)*2*np.pi/n)
            g[6]=g[6]+Pd_planar_diagonal_1_1_0[i,j]            
            f[7]=f[7]+Pd_planar_diagonal_2_2_0[i,j]*np.cos((i-j)*2*np.pi/n)
            g[7]=g[7]+Pd_planar_diagonal_2_2_0[i,j]
            
            f[8]=f[8]+Pd_planar_diagonal_1_2_0[i,j]*np.cos((i-j)*2*np.pi/n)
            g[8]=g[8]+Pd_planar_diagonal_1_2_0[i,j]            
            f[9]=f[9]+Pd_planar_diagonal_2_1_0[i,j]*np.cos((i-j)*2*np.pi/n)
            g[9]=g[9]+Pd_planar_diagonal_2_1_0[i,j]
            
            f[10]=f[10]+Pd_planar_diagonal_1_0_1[i,j]*np.cos((i-j)*2*np.pi/n)
            g[10]=g[10]+Pd_planar_diagonal_1_0_1[i,j]            
            f[11]=f[11]+Pd_planar_diagonal_2_0_2[i,j]*np.cos((i-j)*2*np.pi/n)
            g[11]=g[11]+Pd_planar_diagonal_2_0_2[i,j]
            
            f[12]=f[12]+Pd_planar_diagonal_1_0_2[i,j]*np.cos((i-j)*2*np.pi/n)
            g[12]=g[12]+Pd_planar_diagonal_1_0_2[i,j]            
            f[13]=f[13]+Pd_planar_diagonal_2_0_1[i,j]*np.cos((i-j)*2*np.pi/n)
            g[13]=g[13]+Pd_planar_diagonal_2_0_1[i,j]
            
            f[14]=f[14]+Pd_planar_diagonal_0_1_1[i,j]*np.cos((i-j)*2*np.pi/n)
            g[14]=g[14]+Pd_planar_diagonal_0_1_1[i,j]            
            f[15]=f[15]+Pd_planar_diagonal_0_2_2[i,j]*np.cos((i-j)*2*np.pi/n)
            g[15]=g[15]+Pd_planar_diagonal_0_2_2[i,j]
            
            f[16]=f[16]+Pd_planar_diagonal_0_1_2[i,j]*np.cos((i-j)*2*np.pi/n)
            g[16]=g[16]+Pd_planar_diagonal_0_1_2[i,j]            
            f[17]=f[17]+Pd_planar_diagonal_0_2_1[i,j]*np.cos((i-j)*2*np.pi/n)
            g[17]=g[17]+Pd_planar_diagonal_0_2_1[i,j]
            
            f[18]=f[18]+Pd_volumetric_diagonal_1_1_1[i,j]*np.cos((i-j)*2*np.pi/n)
            g[18]=g[18]+Pd_volumetric_diagonal_1_1_1[i,j]            
            f[19]=f[19]+Pd_volumetric_diagonal_2_2_2[i,j]*np.cos((i-j)*2*np.pi/n)
            g[19]=g[19]+Pd_volumetric_diagonal_2_2_2[i,j]
            
            f[20]=f[20]+Pd_volumetric_diagonal_1_1_2[i,j]*np.cos((i-j)*2*np.pi/n)
            g[20]=g[20]+Pd_volumetric_diagonal_1_1_2[i,j]            
            f[21]=f[21]+Pd_volumetric_diagonal_2_2_1[i,j]*np.cos((i-j)*2*np.pi/n)
            g[21]=g[21]+Pd_volumetric_diagonal_2_2_1[i,j]
            
            f[22]=f[22]+Pd_volumetric_diagonal_1_2_1[i,j]*np.cos((i-j)*2*np.pi/n)
            g[22]=g[22]+Pd_volumetric_diagonal_1_2_1[i,j]            
            f[23]=f[23]+Pd_volumetric_diagonal_2_1_2[i,j]*np.cos((i-j)*2*np.pi/n)
            g[23]=g[23]+Pd_volumetric_diagonal_2_1_2[i,j]
            
            f[24]=f[24]+Pd_volumetric_diagonal_1_2_2[i,j]*np.cos((i-j)*2*np.pi/n)
            g[24]=g[24]+Pd_volumetric_diagonal_1_2_2[i,j]            
            f[25]=f[25]+Pd_volumetric_diagonal_2_1_1[i,j]*np.cos((i-j)*2*np.pi/n)
            g[25]=g[25]+Pd_volumetric_diagonal_2_1_1[i,j]
    
    
    Temp = f / g
    
    Fmax_line_likeness = np.max(Temp)
    
    return Fmax_line_likeness        

def maxlinelikeness_tf(theta,delta,distance,threshold,size):
    theta = theta[0,:,:,:,0]
    delta = delta[0,:,:,:,0]
    z , y , x = theta.shape
    n = size
        
    theta_selected = theta[distance : z - distance,distance : y - distance,distance : x - distance]
    delta_selected = delta[distance : z - distance,distance : y - distance,distance : x - distance]
   
    for m1 in range (n):
      for m2 in range(n):
         cond_1 = tf.greater_equal(theta_selected,2*(m1-1)*np.pi/2/n)
         cond_2 = tf.less(theta_selected,(2*(m1-1)+1)*np.pi/2/n)
         cond_5 = tf.greater_equal(delta_selected,threshold)         

         #Pd_axial_1_0_0         
         theta_1_0_0 = theta[2*distance : z,distance : y - distance,distance : x - distance]
         cond_3 = tf.greater_equal(theta_1_0_0,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_1_0_0,(2*(m1-1)+1)*np.pi/2/n)
         delta_1_0_0 = delta[2*distance : z,distance : y - distance,distance : x - distance]
         cond_6 = tf.less(delta_1_0_0,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_axial_1_0_0 = s
         elif (m2 == 0) :
            Pd_axial_1_0_0 = tf.concat([Pd_axial_1_0_0,s],1)
         else : 
            Pd_axial_1_0_0 = tf.concat([Pd_axial_1_0_0,s],1)
        
         #Pd_axial_2_0_0
         theta_2_0_0 = theta[0 : z - 2*distance,distance : y - distance,distance : x - distance]
         cond_3 = tf.greater_equal(theta_2_0_0,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_2_0_0,(2*(m1-1)+1)*np.pi/2/n)
         delta_2_0_0 = delta[0 : z - 2*distance,distance : y - distance,distance : x - distance]
         cond_6 = tf.less(delta_2_0_0,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_axial_2_0_0 = s
         elif (m2 == 0) :
            Pd_axial_2_0_0 = tf.concat([Pd_axial_2_0_0,s],1)
         else : 
            Pd_axial_2_0_0 = tf.concat([Pd_axial_2_0_0,s],1)
        
         #Pd_axial_0_1_0
         theta_0_1_0 = theta[distance : z - distance,2*distance : y,distance : x - distance]
         cond_3 = tf.greater_equal(theta_0_1_0,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_0_1_0,(2*(m1-1)+1)*np.pi/2/n)
         delta_0_1_0 = delta[distance : z - distance,2*distance : y ,distance : x - distance]
         cond_6 = tf.less(delta_0_1_0,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_axial_0_1_0 = s
         elif (m2 == 0) :
            Pd_axial_0_1_0 = tf.concat([Pd_axial_0_1_0,s],1)
         else : 
            Pd_axial_0_1_0 = tf.concat([Pd_axial_0_1_0,s],1)
         
         #Pd_axial_0_2_0
         theta_0_2_0 = theta[distance : z - distance,0 : y - 2*distance,distance : x - distance]
         cond_3 = tf.greater_equal(theta_0_2_0,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_0_2_0,(2*(m1-1)+1)*np.pi/2/n)
         delta_0_2_0 = delta[distance : z - distance,0 : y - 2*distance,distance : x - distance]
         cond_6 = tf.less(delta_0_2_0,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_axial_0_2_0 = s
         elif (m2 == 0) :
            Pd_axial_0_2_0 = tf.concat([Pd_axial_0_2_0,s],1)
         else : 
            Pd_axial_0_2_0 = tf.concat([Pd_axial_0_2_0,s],1)
                     
         #Pd_axial_0_0_1
         theta_0_0_1 = theta[distance : z - distance,distance : y - distance,2*distance : x]
         cond_3 = tf.greater_equal(theta_0_0_1,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_0_0_1,(2*(m1-1)+1)*np.pi/2/n)
         delta_0_0_1 = delta[distance : z - distance,distance : y - distance,2*distance : x]
         cond_6 = tf.less(delta_0_0_1,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_axial_0_0_1 = s
         elif (m2 == 0) :
            Pd_axial_0_0_1 = tf.concat([Pd_axial_0_0_1,s],1)
         else : 
            Pd_axial_0_0_1 = tf.concat([Pd_axial_0_0_1,s],1)
         
         #Pd_axial_0_0_2
         theta_0_0_2 = theta[distance : z - distance,distance : y - distance,0 : x - 2*distance]
         cond_3 = tf.greater_equal(theta_0_0_2,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_0_0_2,(2*(m1-1)+1)*np.pi/2/n)
         delta_0_0_2 = delta[distance : z - distance,distance : y - distance,0 : x - 2*distance]
         cond_6 = tf.less(delta_0_0_2,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_axial_0_0_2 = s
         elif (m2 == 0) :
            Pd_axial_0_0_2 = tf.concat([Pd_axial_0_0_2,s],1)
         else : 
            Pd_axial_0_0_2 = tf.concat([Pd_axial_0_0_2,s],1)
         
         #Pd_planar_diagonal_1_1_0
         theta_1_1_0 = theta[2*distance : z,2*distance : y,distance : x - distance]
         cond_3 = tf.greater_equal(theta_1_1_0,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_1_1_0,(2*(m1-1)+1)*np.pi/2/n)
         delta_1_1_0 = delta[2*distance : z,2*distance : y,distance : x - distance]
         cond_6 = tf.less(delta_1_1_0,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_1_1_0 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_1_1_0 = tf.concat([Pd_planar_diagonal_1_1_0,s],1)
         else : 
            Pd_planar_diagonal_1_1_0 = tf.concat([Pd_planar_diagonal_1_1_0,s],1)                    
                           
         #Pd_planar_diagonal_2_2_0
         theta_2_2_0 = theta[0 : z - 2*distance,0 : y - 2*distance,distance : x - distance]
         cond_3 = tf.greater_equal(theta_2_2_0,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_2_2_0,(2*(m1-1)+1)*np.pi/2/n)
         delta_2_2_0 = delta[0 : z - 2*distance,0 : y - 2*distance,distance : x - distance]
         cond_6 = tf.less(delta_2_2_0,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_2_2_0 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_2_2_0 = tf.concat([Pd_planar_diagonal_2_2_0,s],1)
         else : 
            Pd_planar_diagonal_2_2_0 = tf.concat([Pd_planar_diagonal_2_2_0,s],1)
         
         #Pd_planar_diagonal_1_2_0
         theta_1_2_0 = theta[2*distance : z,0 : y - 2*distance,distance : x - distance]
         cond_3 = tf.greater_equal(theta_1_2_0,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_1_2_0,(2*(m1-1)+1)*np.pi/2/n)
         delta_1_2_0 = delta[2*distance : z,0 : y - 2*distance,distance : x - distance]
         cond_6 = tf.less(delta_1_2_0,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_1_2_0 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_1_2_0 = tf.concat([Pd_planar_diagonal_1_2_0,s],1)
         else : 
            Pd_planar_diagonal_1_2_0 = tf.concat([Pd_planar_diagonal_1_2_0,s],1)

         #Pd_planar_diagonal_2_1_0
         theta_2_1_0 = theta[0 : z - 2*distance,2*distance : y,distance : x - distance]
         cond_3 = tf.greater_equal(theta_2_1_0,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_2_1_0,(2*(m1-1)+1)*np.pi/2/n)
         delta_2_1_0 = delta[0 : z - 2*distance,2*distance : y,distance : x - distance]
         cond_6 = tf.less(delta_2_1_0,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_2_1_0 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_2_1_0 = tf.concat([Pd_planar_diagonal_2_1_0,s],1)
         else : 
            Pd_planar_diagonal_2_1_0 = tf.concat([Pd_planar_diagonal_2_1_0,s],1)

         #Pd_planar_diagonal_1_0_1
         theta_1_0_1 = theta[2*distance : z,distance : y - distance,2*distance : x]
         cond_3 = tf.greater_equal(theta_1_0_1,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_1_0_1,(2*(m1-1)+1)*np.pi/2/n)
         delta_1_0_1 = delta[2*distance : z,distance : y - distance,2*distance : x]
         cond_6 = tf.less(delta_1_0_1,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_1_0_1 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_1_0_1 = tf.concat([Pd_planar_diagonal_1_0_1,s],1)
         else : 
            Pd_planar_diagonal_1_0_1 = tf.concat([Pd_planar_diagonal_1_0_1,s],1)

         #Pd_planar_diagonal_2_0_2
         theta_2_0_2 = theta[0 : z - 2*distance,distance : y - distance,0 : x - 2*distance]
         cond_3 = tf.greater_equal(theta_2_0_2,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_2_0_2,(2*(m1-1)+1)*np.pi/2/n)
         delta_2_0_2 = delta[0 : z - 2*distance,distance : y - distance,0 : x - 2*distance]
         cond_6 = tf.less(delta_2_0_2,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_2_0_2 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_2_0_2 = tf.concat([Pd_planar_diagonal_2_0_2,s],1)
         else : 
            Pd_planar_diagonal_2_0_2 = tf.concat([Pd_planar_diagonal_2_0_2,s],1)

         #Pd_planar_diagonal_1_0_2
         theta_1_0_2 = theta[2*distance : z,distance : y - distance,0 : x - 2*distance]
         cond_3 = tf.greater_equal(theta_1_0_2,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_1_0_2,(2*(m1-1)+1)*np.pi/2/n)
         delta_1_0_2 = delta[2*distance : z,distance : y - distance,0 : x - 2*distance]
         cond_6 = tf.less(delta_1_0_2,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_1_0_2 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_1_0_2 = tf.concat([Pd_planar_diagonal_1_0_2,s],1)
         else : 
            Pd_planar_diagonal_1_0_2 = tf.concat([Pd_planar_diagonal_1_0_2,s],1)

         #Pd_planar_diagonal_2_0_1
         theta_2_0_1 = theta[0 : z - 2*distance,distance : y - distance,2*distance : x]
         cond_3 = tf.greater_equal(theta_2_0_1,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_2_0_1,(2*(m1-1)+1)*np.pi/2/n)
         delta_2_0_1 = delta[0 : z - 2*distance,distance : y - distance,2*distance : x]
         cond_6 = tf.less(delta_2_0_1,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_2_0_1 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_2_0_1 = tf.concat([Pd_planar_diagonal_2_0_1,s],1)
         else : 
            Pd_planar_diagonal_2_0_1 = tf.concat([Pd_planar_diagonal_2_0_1,s],1)

         #Pd_planar_diagonal_0_1_1
         theta_0_1_1 = theta[distance : z - distance,2*distance : y,2*distance : x]
         cond_3 = tf.greater_equal(theta_0_1_1,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_0_1_1,(2*(m1-1)+1)*np.pi/2/n)
         delta_0_1_1 = delta[distance : z - distance,2*distance : y,2*distance : x]
         cond_6 = tf.less(delta_0_1_1,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_0_1_1 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_0_1_1 = tf.concat([Pd_planar_diagonal_0_1_1,s],1)
         else : 
            Pd_planar_diagonal_0_1_1 = tf.concat([Pd_planar_diagonal_0_1_1,s],1)

         #Pd_planar_diagonal_0_2_2
         theta_0_2_2 = theta[distance : z - distance,0 : y - 2*distance,0 : x - 2*distance]
         cond_3 = tf.greater_equal(theta_0_2_2,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_0_2_2,(2*(m1-1)+1)*np.pi/2/n)
         delta_0_2_2 = delta[distance : z - distance,0 : y - 2*distance,0 : x - 2*distance]
         cond_6 = tf.less(delta_0_2_2,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_0_2_2 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_0_2_2 = tf.concat([Pd_planar_diagonal_0_2_2,s],1)
         else : 
            Pd_planar_diagonal_0_2_2 = tf.concat([Pd_planar_diagonal_0_2_2,s],1)

         #Pd_planar_diagonal_0_1_2
         theta_0_1_2 = theta[distance : z - distance,2*distance : y,0 : x - 2*distance]
         cond_3 = tf.greater_equal(theta_0_1_2,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_0_1_2,(2*(m1-1)+1)*np.pi/2/n)
         delta_0_1_2 = delta[distance : z - distance,2*distance : y,0 : x - 2*distance]
         cond_6 = tf.less(delta_0_1_2,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_0_1_2 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_0_1_2 = tf.concat([Pd_planar_diagonal_0_1_2,s],1)
         else : 
            Pd_planar_diagonal_0_1_2 = tf.concat([Pd_planar_diagonal_0_1_2,s],1)

         #Pd_planar_diagonal_0_2_1
         theta_0_2_1 = theta[distance : z - distance,0 : y - 2*distance,2*distance : x]
         cond_3 = tf.greater_equal(theta_0_2_1,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_0_2_1,(2*(m1-1)+1)*np.pi/2/n)
         delta_0_2_1 = delta[distance : z - distance,0 : y - 2*distance,2*distance : x]
         cond_6 = tf.less(delta_0_2_1,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_planar_diagonal_0_2_1 = s
         elif (m2 == 0) :
            Pd_planar_diagonal_0_2_1 = tf.concat([Pd_planar_diagonal_0_2_1,s],1)
         else : 
            Pd_planar_diagonal_0_2_1 = tf.concat([Pd_planar_diagonal_0_2_1,s],1)
         
         #Pd_volumetric_diagonal_1_1_1
         theta_1_1_1 = theta[2*distance : z,2*distance : y,2*distance : x]
         cond_3 = tf.greater_equal(theta_1_1_1,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_1_1_1,(2*(m1-1)+1)*np.pi/2/n)
         delta_1_1_1 = delta[2*distance : z,2*distance : y,2*distance : x]
         cond_6 = tf.less(delta_1_1_1,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_volumetric_diagonal_1_1_1 = s
         elif (m2 == 0) :
            Pd_volumetric_diagonal_1_1_1 = tf.concat([Pd_volumetric_diagonal_1_1_1,s],1)
         else : 
            Pd_volumetric_diagonal_1_1_1 = tf.concat([Pd_volumetric_diagonal_1_1_1,s],1)

         #Pd_volumetric_diagonal_2_2_2
         theta_2_2_2 = theta[0 : z - 2*distance,0 : y - 2*distance,0 : x - 2*distance]
         cond_3 = tf.greater_equal(theta_2_2_2,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_2_2_2,(2*(m1-1)+1)*np.pi/2/n)
         delta_2_2_2 = delta[0 : z - 2*distance,0 : y - 2*distance,0 : x - 2*distance]
         cond_6 = tf.less(delta_2_2_2,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_volumetric_diagonal_2_2_2 = s
         elif (m2 == 0) :
            Pd_volumetric_diagonal_2_2_2 = tf.concat([Pd_volumetric_diagonal_2_2_2,s],1)
         else : 
            Pd_volumetric_diagonal_2_2_2 = tf.concat([Pd_volumetric_diagonal_2_2_2,s],1)
         
         #Pd_volumetric_diagonal_1_1_2
         theta_1_1_2 = theta[2*distance : z,2*distance : y,0 : x - 2*distance]
         cond_3 = tf.greater_equal(theta_1_1_2,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_1_1_2,(2*(m1-1)+1)*np.pi/2/n)
         delta_1_1_2 = delta[2*distance : z,2*distance : y,0 : x - 2*distance]
         cond_6 = tf.less(delta_1_1_2,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_volumetric_diagonal_1_1_2 = s
         elif (m2 == 0) :
            Pd_volumetric_diagonal_1_1_2 = tf.concat([Pd_volumetric_diagonal_1_1_2,s],1)
         else : 
            Pd_volumetric_diagonal_1_1_2 = tf.concat([Pd_volumetric_diagonal_1_1_2,s],1)
         
         #Pd_volumetric_diagonal_2_2_1
         theta_2_2_1 = theta[0 : z - 2*distance,0 : y - 2*distance,2*distance : x]
         cond_3 = tf.greater_equal(theta_2_2_1,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_2_2_1,(2*(m1-1)+1)*np.pi/2/n)
         delta_2_2_1 = delta[0 : z - 2*distance,0 : y - 2*distance,2*distance : x]
         cond_6 = tf.less(delta_2_2_1,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_volumetric_diagonal_2_2_1 = s
         elif (m2 == 0) :
            Pd_volumetric_diagonal_2_2_1 = tf.concat([Pd_volumetric_diagonal_2_2_1,s],1)
         else : 
            Pd_volumetric_diagonal_2_2_1 = tf.concat([Pd_volumetric_diagonal_2_2_1,s],1)
         
         #Pd_volumetric_diagonal_1_2_1
         theta_1_2_1 = theta[2*distance : z,0 : y - 2*distance,2*distance : x]
         cond_3 = tf.greater_equal(theta_1_2_1,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_1_2_1,(2*(m1-1)+1)*np.pi/2/n)
         delta_1_2_1 = delta[2*distance : z,0 : y - 2*distance,2*distance : x]
         cond_6 = tf.less(delta_1_2_1,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_volumetric_diagonal_1_2_1 = s
         elif (m2 == 0) :
            Pd_volumetric_diagonal_1_2_1 = tf.concat([Pd_volumetric_diagonal_1_2_1,s],1)
         else : 
            Pd_volumetric_diagonal_1_2_1 = tf.concat([Pd_volumetric_diagonal_1_2_1,s],1)

         #Pd_volumetric_diagonal_2_1_2
         theta_2_1_2 = theta[0 : z - 2*distance,2*distance : y,0 : x - 2*distance]
         cond_3 = tf.greater_equal(theta_2_1_2,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_2_1_2,(2*(m1-1)+1)*np.pi/2/n)
         delta_2_1_2 = delta[0 : z - 2*distance,2*distance : y,0 : x - 2*distance]
         cond_6 = tf.less(delta_2_1_2,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_volumetric_diagonal_2_1_2 = s
         elif (m2 == 0) :
            Pd_volumetric_diagonal_2_1_2 = tf.concat([Pd_volumetric_diagonal_2_1_2,s],1)
         else : 
            Pd_volumetric_diagonal_2_1_2 = tf.concat([Pd_volumetric_diagonal_2_1_2,s],1)

         #Pd_volumetric_diagonal_1_2_2
         theta_1_2_2 = theta[2*distance : z,0 : y - 2*distance,0 : x - 2*distance]
         cond_3 = tf.greater_equal(theta_1_2_2,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_1_2_2,(2*(m1-1)+1)*np.pi/2/n)
         delta_1_2_2 = delta[2*distance : z,0 : y - 2*distance,0 : x - 2*distance]
         cond_6 = tf.less(delta_1_2_2,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_volumetric_diagonal_1_2_2 = s
         elif (m2 == 0) :
            Pd_volumetric_diagonal_1_2_2 = tf.concat([Pd_volumetric_diagonal_1_2_2,s],1)
         else : 
            Pd_volumetric_diagonal_1_2_2 = tf.concat([Pd_volumetric_diagonal_1_2_2,s],1)

         #Pd_volumetric_diagonal_2_1_1
         theta_2_1_1 = theta[0 : z - 2*distance,2*distance : y,2*distance : x]
         cond_3 = tf.greater_equal(theta_2_1_1,2*(m2-1)*np.pi/2/n)
         cond_4 = tf.less(theta_2_1_1,(2*(m1-1)+1)*np.pi/2/n)
         delta_2_1_1 = delta[0 : z - 2*distance,2*distance : y,2*distance : x]
         cond_6 = tf.less(delta_2_1_1,threshold)
         cond = tf.logical_and(cond_1,cond_2)
         cond = tf.logical_and(cond_3,cond)
         cond = tf.logical_and(cond_4,cond)
         cond = tf.logical_and(cond_5,cond)
         cond = tf.logical_and(cond_6,cond)
         s = tf.reshape(tf.reduce_sum(tf.cast(cond, tf.int32)),[1,1])
         if (m1 == 0 and m2 == 0):
            Pd_volumetric_diagonal_2_1_1 = s
         elif (m2 == 0) :
            Pd_volumetric_diagonal_2_1_1 = tf.concat([Pd_volumetric_diagonal_2_1_1,s],1)
         else : 
            Pd_volumetric_diagonal_2_1_1 = tf.concat([Pd_volumetric_diagonal_2_1_1,s],1)
    
    
    indices_i, _ = tf.meshgrid(tf.range(n),tf.range(n), indexing='ij')
    indices_j, _ = tf.meshgrid(tf.range(n),tf.range(n), indexing='xy')
    tensor_sub = tf.subtract(indices_i,indices_j)    
    tensor_cos = tf.multiply(tf.cast(tensor_sub,tf.float32),2*np.pi/n)   
    tensor_cos = tf.cos(tensor_cos) 
        
    f = sum_and_reshape(Pd_axial_1_0_0,tensor_cos,n)
    g = tf.reshape(tf.reduce_sum(Pd_axial_1_0_0),[1])            
   
    
    f = tf.concat([f,sum_and_reshape(Pd_axial_2_0_0,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_axial_2_0_0),[1])],0)

    f = tf.concat([f,sum_and_reshape(Pd_axial_0_1_0,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_axial_0_1_0),[1])],0)

    f = tf.concat([f,sum_and_reshape(Pd_axial_0_2_0,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_axial_0_2_0),[1])],0) 

    f = tf.concat([f,sum_and_reshape(Pd_axial_0_0_1,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_axial_0_0_1),[1])],0)

    f = tf.concat([f,sum_and_reshape(Pd_axial_0_0_2,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_axial_0_0_2),[1])],0)

    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_1_1_0,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_1_1_0),[1])],0) 

    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_2_2_0,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_2_2_0),[1])],0)   
    
    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_1_2_0,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_1_2_0),[1])],0)
    
    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_2_1_0,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_2_1_0),[1])],0)
   
    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_1_0_1,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_1_0_1),[1])],0)

    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_2_0_2,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_2_0_2),[1])],0)

    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_1_0_2,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_1_0_2),[1])],0)

    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_2_0_1,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_2_0_1),[1])],0)
            
    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_0_1_1,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_0_1_1),[1])],0)       
            
    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_0_2_2,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_0_2_2),[1])],0)           
    
    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_0_1_2,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_0_1_2),[1])],0)
           
    f = tf.concat([f,sum_and_reshape(Pd_planar_diagonal_0_2_1,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_planar_diagonal_0_2_1),[1])],0)        
                       
    f = tf.concat([f,sum_and_reshape(Pd_volumetric_diagonal_1_1_1,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_volumetric_diagonal_1_1_1),[1])],0)
    
    f = tf.concat([f,sum_and_reshape(Pd_volumetric_diagonal_2_2_2,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_volumetric_diagonal_2_2_2),[1])],0)
                     
    f = tf.concat([f,sum_and_reshape(Pd_volumetric_diagonal_1_1_2,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_volumetric_diagonal_1_1_2),[1])],0)   
            
    f = tf.concat([f,sum_and_reshape(Pd_volumetric_diagonal_2_2_1,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_volumetric_diagonal_2_2_1),[1])],0)

    f = tf.concat([f,sum_and_reshape(Pd_volumetric_diagonal_1_2_1,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_volumetric_diagonal_1_2_1),[1])],0)        

    f = tf.concat([f,sum_and_reshape(Pd_volumetric_diagonal_2_1_2,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_volumetric_diagonal_2_1_2),[1])],0)

    f = tf.concat([f,sum_and_reshape(Pd_volumetric_diagonal_1_2_2,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_volumetric_diagonal_1_2_2),[1])],0)
   
    f = tf.concat([f,sum_and_reshape(Pd_volumetric_diagonal_2_1_1,tensor_cos,n)],0)
    g = tf.concat([g,tf.reshape(tf.reduce_sum(Pd_volumetric_diagonal_2_1_1),[1])],0)       
               
    
    Temp = tf.divide(f,tf.cast(g,tf.float32))    

    Fmax_line_likeness = tf.reshape(tf.reduce_max(Temp),[1])
    
    return Fmax_line_likeness        
  

def standard_deviation(array):
     v = np.var(array)
     std = np.power(v, 0.5)
     return std            

# 3D convolution
@tf.function
def conv3d(x, W, stride=1):
    conv_3d = tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding='SAME')
    return conv_3d            

@tf.function
def argmax_tesor(tensor):
    max_val = tf.reduce_max(tensor)
    indices = tf.where(tf.equal(tensor, max_val))
    return tf.reduce_max(indices,0),max_val

def calculate_theta(delta_1,delta_2):
    zero1 = tf.equal(delta_1,0)
    zero2 = tf.equal(delta_2,0)
    zeroBoth12 = tf.logical_and(zero1, zero2)
    delta2_one = tf.add(delta_2,1)
    delta2_pi = tf.add(delta_2, np.pi)  
    delta_2 = tf.where(zero2,delta2_one,delta_2)    
    
    theta12 = tf.atan(tf.math.divide(delta_1, delta_2)) + np.pi / 2.0
    theta12 = tf.where(zero2, delta2_pi, theta12)
    theta12 = tf.where(zeroBoth12,delta_2, theta12)
    
    return theta12

def histogram_direction(theta,deltaG):
    n = 16
    t = 12
    for ni in range(n):
        cond1 = tf.greater_equal(deltaG,t)
        cond2 = tf.greater_equal(theta,(2*ni-1) * np.pi / (2 * n))
        cond3 = tf.less(theta,(2*ni+1) * np.pi / (2*n))
        cond = tf.logical_and(cond1, cond2)
        cond = tf.logical_and(cond3,cond)
        s = tf.reshape(tf.reduce_sum(tf.cast(cond,tf.int32)),[1])
        if (ni == 0):
          hd = s
        else:
          hd = tf.concat([hd,s],0)
    
    hd = tf.cast(hd,tf.float32)
    hd = tf.math.divide(hd,tf.reduce_sum(hd))
    hd_max_index = tf.cast((tf.argmax(hd)),tf.int32)
    r = tf.cast((tf.range(0,n,1)),tf.int32)
    rp = tf.pow(tf.subtract(r,hd_max_index),2)
    hd_x = tf.expand_dims(hd,1)
    rp_x = tf.cast((tf.transpose(tf.expand_dims(rp,1))),tf.float32)
    fdir = tf.squeeze(tf.matmul(rp_x,hd_x))
    return fdir

def sum_and_reshape(tensor_1,tensor_2,size):
   return tf.reshape(tf.reduce_sum(tf.multiply(tf.reshape(tf.cast(tensor_1,tf.float32),[size,size]),tensor_2)),[1])

def split_sub_tensors(tensor):
   # Split dim 0
   splits1 = tf.split(tensor, 2, axis=0) 
   # Split dim 1 for each
   splits2 = [tf.split(s, 2, axis=1) for s in splits1] 
   # Split dim 2 for each
   final_sub_tensors = []
   for s2 in splits2:
      for s3 in s2:
         final_sub_tensors.extend(tf.split(s3, 2, axis=2))
   # final_sub_tensors contains 8 tensors of (24,24,24)
   
   return final_sub_tensors   



           
    
    
    
            
            
           
    
    
    
    
    
            
            
             
     
        
    
    
