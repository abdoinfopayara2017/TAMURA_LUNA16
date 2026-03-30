
import numpy as np

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

def standard_deviation(array):
     v = np.var(array)
     std = np.power(v, 0.5)
     return std            
            
            
           
    
    
    
            
            
           
    
    
    
    
    
            
            
             
     
        
    
    
