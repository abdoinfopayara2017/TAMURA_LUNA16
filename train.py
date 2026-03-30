
import numpy as np
import pandas as pd
from dataprocess.textural_features import calc_coarseness,contrast,directionality,linelikeness,regularity,roughness

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
    tab = np.zeros(0)
    for i in range (1):
     image = np.load(images[i][0])
     if np.sum(image):
      #F_crs = calc_coarseness(image,5,0.9)     
      #F_cos = contrast(image)
      #F_dir = directionality(image)
      #F_lin = linelikeness(image)
      F_reg = regularity(image)
      #F_rgh= roughness(image)
      tab = np.append(tab,F_reg)
    
    print("Median Max and Min of dataset %d : %2f %2f %2f" %(truth,np.median(tab),np.max(tab),np.min(tab)))


train('dataprocess\\data\\training_class_0.csv',0)
#train('dataprocess\\data\\training_class_1.csv',1)