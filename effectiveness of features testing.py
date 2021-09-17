# Effectiveness of features testing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import *
from sklearn import tree

data=pd.read_csv('final_data_10.csv')

# Define x,y
y = data.clk_or_not
# Basic features (user_profile+ad features)
x = data.drop(['clk_or_not','nonclk','clk','1day_pv','1day_cart','1day_fav','1day_buy',
               '3day_pv','3day_cart','3day_fav','3day_buy','7day_pv','7day_cart','7day_fav',
                '7day_buy','15day_pv','15day_cart','15day_fav','15day_buy','his_ctr'], axis=1)

# Basic features + historical CTR feature
# x = data.drop(['clk_or_not','nonclk','clk','1day_pv','1day_cart','1day_fav','1day_buy',
#                '3day_pv','3day_cart','3day_fav','3day_buy','7day_pv','7day_cart','7day_fav',
#                 '7day_buy','15day_pv','15day_cart','15day_fav','15day_buy'], axis=1)

# Basic features + user behavioral features
# x = data.drop(['clk_or_not','nonclk','clk','his_ctr'], axis=1)

# Basic features + user behavioral features + historical CTR feature
# x = data.drop(['clk_or_not','nonclk','clk'], axis=1)
    
# Data spliting
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)


# Metrics
print("Accuracy:",accuracy_score(Y_test, Y_pred))
print("F1 score:",f1_score(Y_test,Y_pred))
print("AUC:",roc_auc_score(Y_test,Y_pred))
print("\n")  
