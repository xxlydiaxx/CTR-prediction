import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


y = data.clk_or_not
x = data.drop(['clk_or_not','nonclk','clk'], axis=1) 
    
# Identify more relevant features by RandomForestRegressor()
# This function is from our group's IMA01
model=RandomForestRegressor()
model.fit(x,y)
    
features = x.columns
importances = model.feature_importances_
indices = np.argsort(importances[0:24])  # Top 25 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
    
# Remove unimportant features
# newdata = data.drop(['asin','series_sum'],axis=1)
# print("Successfully remove unimportant features.")
