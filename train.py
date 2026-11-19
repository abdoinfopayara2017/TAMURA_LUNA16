
import numpy as np
import pandas as pd
import dataprocess.textural_features as txnp 
import tensorflow as tf
from datetime import datetime

#@tf.function
def train(path):
    
    
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
      for index in range (1) : #len(csvimagedata) :        
        image = np.load(images[index][0])       
        if np.sum(image):          
          print("Current time F_crs :" , datetime.now())
          F_crs = txnp.calc_coarseness(image,5,0.9)          
          csvimagedata.loc[index, "coarseness"] = float(F_crs)             
          print("Current time F_cos :" , datetime.now())
          F_cos = txnp.contrast(image)      
          csvimagedata.loc[index, "contrast"] = float(F_cos)          
          print("Current time F_dir :" , datetime.now())
          F_dir,*_ = txnp.directionality(image)
          csvimagedata.loc[index, "directionality"] = float(F_dir)      
          print(datetime.now())
          #F_lin_tf = txnp.linelikeness(image)
          #csvimagedata.loc[index, "linelikeness"] = float(F_lin_tf)
          #F_reg = txnp.regularity(image)
          #csvimagedata.loc[index, "regularity"] = float(F_reg)
          #F_rgh= txnp.roughness(F_crs,F_cos)  
          #csvimagedata.loc[index, "roughness"] = float(F_rgh)
        print('record number %d is done from %d'% (index,csvimagedata.index.size))
                
      
      csvimagedata.to_csv("data.csv", index=False)    

train('dataprocess/data/training_origine.csv')