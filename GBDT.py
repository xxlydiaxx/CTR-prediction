import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve


data=pd.read_csv('test_data.csv')

y = data.clk_or_not
X = data.drop(['clk_or_not','nonclk','clk'], axis=1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

#print(X_train.shape)  (20000, 20)
#print(y_train.shape) (20000,)

gbdt = GradientBoostingClassifier(n_estimators=5)

"""
n_estimators,最大的弱学习器的个数，即有多少个回归树
max_depth : int, default=3。每个回归树的的深度
"""

gbdt_enc = OneHotEncoder()
lr = LogisticRegression(max_iter=1000)
gbdt.fit(X_train, y_train) # 训练GBDT模型

gbdt_enc.fit(gbdt.apply(X_train)[:, :, 0]) # one-hot

lr.fit(gbdt_enc.transform(gbdt.apply(X_train_lr)[:, :, 0]), y_train_lr) # train model

y_pred_gbdt_lr = lr.predict_proba(
    gbdt_enc.transform(gbdt.apply(X_test)[:, :, 0]))[:, 1]
print(y_pred_gbdt_lr)
fpr_grd_lr, tpr_grd_lr, _ = roc_curve(y_test, y_pred_gbdt_lr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_grd_lr, tpr_grd_lr, label='GBDT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_grd_lr, tpr_grd_lr, label='GBDT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
