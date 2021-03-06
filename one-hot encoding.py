# one-hot encoding
import pandas as pd

df_1=pd.read_csv('user_profile.csv')
df_1=df_1.drop(df_1.columns[[0,1]], axis=1) 
df_1.loc[df_1['final_gender_code']==1,'final_gender_code'] = 'male'
df_1.loc[df_1['final_gender_code']==2,'final_gender_code'] = 'female'
df_1.loc[df_1['occupation']==1,'occupation'] = 'is_student'
df_1.loc[df_1['occupation']==0,'occupation'] = 'not_student'

# Transform feature gender and degree using one-hot-encoding

df_2=pd.get_dummies(df_1, columns=['final_gender_code','occupation'] ) 
df_new=pd.concat([df_1,df_2],axis=1) 
df_new=df_new.drop(['final_gender_code','occupation'], axis=1) 


df_new.head(10)
# df_new.to_csv("user_profile2.csv")


