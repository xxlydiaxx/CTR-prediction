#User Behavioural Features Constructing

import pandas as pd


time_1=86400
time_2=259200
time_3=604800
time_4=1296000

df_raw=pd.read_csv('raw_sample.csv')
df_log=pd.read_csv('behavior_log.csv')
df_fea=pd.read_csv('ad_feature.csv')
df_fea=df_fea.rename(columns={'cate_id':'cate'})

#raw_sample and ad_featuer -- group by adgroup to get brand and cate
df_raw=pd.merge(df_raw,df_fea[['adgroup_id','brand','cate']],on='adgroup_id')
#raw_sample - endtime
df_raw=df_raw.rename({'time_stamp':'time_end'},axis='columns')
#behavior_log - starttime
df_log=df_log.rename({'time_stamp':'time_start'},axis='columns')
#merge raw_sample and behavior_log group by user,brand,cate
df_raw=pd.merge(df_raw,df_log[['user','brand','cate','time_start','btag']],on=['user','brand','cate'])
#calculate time gap
df_raw['time_span']=df_raw['time_end']-df_raw['time_start']
#drop columns
df_raw=df_raw.drop(['time_end','adgroup_id','pid','nonclk','clk','time_start'],1)

#compare time gap
df_out1=df_raw.loc[(df_raw.time_span>0)&(df_raw.time_span<time_1),]
df_out2=df_raw.loc[(df_raw.time_span>0)&(df_raw.time_span<time_2),]
df_out3=df_raw.loc[(df_raw.time_span>0)&(df_raw.time_span<time_3),]
df_out4=df_raw.loc[(df_raw.time_span>0)&(df_raw.time_span<time_4),]


#count each btag group by user,brand,cate
count1=df_out1.groupby(['user','brand','cate']).btag.value_counts()
output1=count1.to_frame()
output1.columns=['values']
output1.reset_index(inplace=True)

count2=df_out2.groupby(['user','brand','cate']).btag.value_counts()
output2=count2.to_frame()
output2.columns=['values']
output2.reset_index(inplace=True)

count3=df_out3.groupby(['user','brand','cate']).btag.value_counts()
output3=count3.to_frame()
output3.columns=['values']
output3.reset_index(inplace=True)

count4=df_out4.groupby(['user','brand','cate']).btag.value_counts()
output4=count4.to_frame()
output4.columns=['values']
output4.reset_index(inplace=True)

output1.to_csv('raw_sample_1.csv')
output2.to_csv('raw_sample_2.csv')
output3.to_csv('raw_sample_3.csv')
output4.to_csv('raw_sample_4.csv')
