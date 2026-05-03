
import numpy as np
import pandas as pd
import dataprocess.textural_features as txnp
import dataprocess.textural_features_tensor_flow as txtf 
import tensorflow as tf
import tensorflow_probability as tfp

def train(path,truth):
    
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
    np.random.shuffle(data)
    # For Image
    images = data[:, 1:]
    # For Labels
    labels = data[:, 0]   
    tab = tf.zeros(0)
    tabnp = np.zeros(0)
    for i in range (1):
     image = np.load(images[i][0])
     if np.sum(image):
      
      #F_crs = txtf
      #F_crs_np = txnp.calc_coarseness(image,5,0.9)       
      #F_cos = txtf.contrast(image)
      #F_cos_np = txnp.contrast(image)      
      #F_dir,*_ = txtf.directionality(image)
      #F_dir_np,*_ = txnp.directionality(image)
      #F_lin = txnp.linelikeness(image)
      #F_lin_tf = txtf.linelikeness(image)
      #F_reg = txtf.regularity(image)
      F_rgh= txtf.roughness(image)
      
      tab = tf.concat([tab,F_rgh],0)
      #tabnp=np.append(tabnp,F_lin)
      
    
    
    tf.print(tfp.stats.percentile(tab, 50.0, interpolation='midpoint'),tf.reduce_max(tab),tf.reduce_min(tab))
    #print("Median Max and Min of dataset %d : %2f %2f %2f" %(truth,np.median(tabnp),np.max(tabnp),np.min(tabnp)))

train('dataprocess\\data\\training_class_0.csv',0)
#train('dataprocess\\data\\training_class_1.csv',1)