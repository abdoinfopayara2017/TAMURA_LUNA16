
import numpy as np
import pandas as pd
import dataprocess.textural_features_tensor_flow as txtf 
import tensorflow as tf

@tf.function
def train(path):
    
    with tf.device('/gpu:0'):
      '''
      Preprocessing for dataset
      '''
      # Read  data set (Train data from CSV file)
      # csv file should have the type:
      # label,data_npy
      # label,data_npy
      # ....
      #

      csvimagedata = pd.read_csv(path)
      data = csvimagedata.iloc[:, :].values
      #np.random.shuffle(data)
      # For Image
      images = data[:, 1:]
      # For Labels
      labels = data[:, 0]    
      tamuraMethod = txtf.TamuraMethod(48,48,48)
      img_shape = (48, 48, 48) 
      image_buffer = np.empty(img_shape, dtype=np.float32)
      for index in range (1) : #csvimagedata.index :        
        mmap_array = np.load(images[index][0],mmap_mode='r')
        image_buffer[:] = mmap_array
        if np.sum(image_buffer):          
          F_crs = tamuraMethod.calc_coarseness(image_buffer,5,0.9)
          csvimagedata.loc[index, "coarseness"] = float(F_crs)             
          #F_cos = txtf.contrast(image)      
          #csvimagedata.loc[index, "contrast"] = float(F_cos)          
          #F_dir,*_ = txtf.directionality(image)
          #csvimagedata.loc[index, "directionality"] = float(F_dir)      
          #F_lin_tf = txtf.linelikeness(image)
          #csvimagedata.loc[index, "linelikeness"] = float(F_lin_tf)
          #F_reg = txtf.regularity(image)
          #csvimagedata.loc[index, "regularity"] = float(F_reg)
          #F_rgh= txtf.roughness(F_crs,F_cos)  
          #csvimagedata.loc[index, "roughness"] = float(F_rgh)'''
        print('record number %d is done from %d'% (index,csvimagedata.index.size))
                
      
      csvimagedata.to_csv("data.csv", index=False)    

train('dataprocess/data/training_origine.csv')