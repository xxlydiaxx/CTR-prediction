# Feature scaling
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


data=pd.read_csv('final_data.csv')

df_features = data.drop(['clk_or_not','nonclk','clk'],axis=1)
    
# Use StandarScaler to standardise and transform 
df_scale = StandardScaler().fit(data[df_features.columns]) 
data[df_features.columns] = df_scale.transform(data[df_features.columns])
    
print('Successfully.')
data.to_csv('data_scale.csv')
