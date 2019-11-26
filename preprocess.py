#!/usr/bin/env python
# coding: utf-8

# # 라이온킹 전처리 코드

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


# # 1. Train

# In[2]:


label = pd.read_csv('../raw/train_label.csv')
activity = pd.read_csv('../raw/train_activity.csv')
payment = pd.read_csv('../raw/train_payment.csv')
trade = pd.read_csv('../raw/train_trade.csv')
pledge = pd.read_csv('../raw/train_pledge.csv')
combat = pd.read_csv('../raw/train_combat.csv')


# ### 1-1. activity

# In[3]:


activity_og = activity.copy()


# In[4]:


def week(x):
    if 1 <= x and x<8:
        return 1
    elif 8 <= x and x<15:
        return 2
    elif 15 <= x and x<22:
        return 3
    else:
        return 4


# In[5]:


temp = activity.pivot_table(index='acc_id', values='day', aggfunc='min').reset_index()
temp['week'] = temp['day'].apply(week)
temp.columns = ['acc_id', '최초접속일', '최초접속주차']

activity['week'] = activity['day'].apply(week)
activity_week1 = activity[activity['week']==1]
activity_week2 = activity[activity['week']==2]
activity_week3 = activity[activity['week']==3]
activity_week4 = activity[activity['week']==4]

첫주차접속일 = activity_week1.pivot_table(index='acc_id', values='day', aggfunc=pd.Series.nunique).reset_index()
첫주차접속일.columns = ['acc_id', '첫주차접속일']
사주차접속일 = activity_week4.pivot_table(index='acc_id', values='day', aggfunc=pd.Series.nunique).reset_index()
사주차접속일.columns = ['acc_id', '사주차접속일']

temp = pd.merge(temp, 첫주차접속일, on='acc_id', how='left')
temp = pd.merge(temp, 사주차접속일, on='acc_id', how='left')
temp.fillna(0, inplace=True)

# unique count - 캐릭터별 최고 접속일
캐릭터별총접속일 = activity.pivot_table(index='acc_id', values='day', aggfunc='count').reset_index()
캐릭터별총접속일.columns = ['acc_id', '캐릭터별총접속일']
temp = pd.merge(temp, 캐릭터별총접속일, on='acc_id', how='left')

# unique count - 캐릭터별 최고 접속일
ID별총접속일 = activity.pivot_table(index='acc_id', values='day', aggfunc=pd.Series.nunique).reset_index()
ID별총접속일.columns = ['acc_id', 'ID별총접속일']
temp = pd.merge(temp, ID별총접속일, on='acc_id', how='left')

temp['최초접속이후비접속횟수'] = 28 - temp['최초접속일'] + 1 - temp['ID별총접속일']

캐릭터별접속일 = activity.pivot_table(index=['acc_id', 'char_id'], values='day', aggfunc=pd.Series.nunique).reset_index()
접속_28일 = 캐릭터별접속일[캐릭터별접속일['day']==28]
접속_28일캐릭터수 = 접속_28일.pivot_table(index='acc_id', values='char_id', aggfunc='count').reset_index()
접속_28일캐릭터수.columns = ['acc_id', '접속28일캐릭터수']
temp = pd.merge(temp, 접속_28일캐릭터수, on='acc_id', how='left')
temp.fillna(0, inplace=True)


# In[6]:


첫주플레이시간평균 = activity_week1.pivot_table(index='acc_id', values='playtime', aggfunc='mean').reset_index()
첫주플레이시간평균.columns = ['acc_id', '첫주플레이시간평균']
사주플레이시간평균 = activity_week4.pivot_table(index='acc_id', values='playtime', aggfunc='mean').reset_index()
사주플레이시간평균.columns = ['acc_id', '사주플레이시간평균']

temp = pd.merge(temp, 첫주플레이시간평균, on='acc_id', how='left')
temp = pd.merge(temp, 사주플레이시간평균, on='acc_id', how='left')
temp.fillna(0, inplace=True)


# In[7]:


activity = activity.drop(['char_id', 'server', 'week'], axis=1)


# In[8]:


activity_day_sum = activity.pivot_table(index=['day','acc_id'], aggfunc='sum').reset_index()
activity_day_sum.fillna(0,inplace=True)


# In[9]:


activity_day_sum_columns = ['day', 'acc_id']
for col in activity_day_sum.columns[2:]:
    activity_day_sum_columns.append(col+'_sum')
activity_day_sum.columns = activity_day_sum_columns


# ### 1-2. payment

# In[10]:


payment_day_sum = payment.pivot_table(index=['day','acc_id'], aggfunc='sum').reset_index()
payment_day_sum.fillna(0, inplace=True)


# In[11]:


payment_day_sum_columns = ['day', 'acc_id']
for col in payment_day_sum.columns[2:]:
    payment_day_sum_columns.append(col+'_sum')
payment_day_sum.columns = payment_day_sum_columns


# In[12]:


payment_day_std = payment.pivot_table(index=['acc_id'], aggfunc='std').reset_index()
payment_day_std.columns = ['acc_id', 'amount_spent_std', 'payment_day_std']


# In[13]:


최고지불금액 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc=max).reset_index()
최고지불금액.columns = ['acc_id', '최고지불금액']
temp = pd.merge(temp, 최고지불금액, on='acc_id', how='left')


# In[14]:


지불금액총합 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc=sum).reset_index()
지불금액총합.columns = ['acc_id', '지불금액총합']
temp = pd.merge(temp, 지불금액총합, on='acc_id', how='left')


# In[15]:


지불횟수 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc=pd.Series.nunique).reset_index()
지불횟수.columns = ['acc_id', '지불횟수']
temp = pd.merge(temp, 지불횟수, on='acc_id', how='left')


# In[16]:


평균지불금액 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc='mean').reset_index()
평균지불금액.columns = ['acc_id', '총평균지불금액']
temp = pd.merge(temp, 평균지불금액, on='acc_id', how='left')


# ### 1-3. trade

# In[17]:


# 일별 개인상점 횟수 및 교환창 횟수 (소스/타겟)
타입횟수소스 = trade.pivot_table(index=['source_acc_id', 'type'], values='item_type', aggfunc='count').reset_index()
개인상점횟수소스 = 타입횟수소스[타입횟수소스['type']==1]
개인상점횟수소스 = 개인상점횟수소스.drop('type', axis=1)           
개인상점횟수소스.columns=['acc_id', '개인상점소스']
교환창횟수소스 = 타입횟수소스[타입횟수소스['type']==0]
교환창횟수소스 = 교환창횟수소스.drop('type', axis=1)           
교환창횟수소스.columns=['acc_id', '교환창소스']

타입횟수타겟 = trade.pivot_table(index=['target_acc_id', 'type'], values='item_type', aggfunc='count').reset_index()
개인상점횟수타겟 = 타입횟수타겟[타입횟수타겟['type']==1]
개인상점횟수타겟 = 개인상점횟수타겟.drop('type', axis=1)           
개인상점횟수타겟.columns=['acc_id', '개인상점타겟']
교환창횟수타겟 = 타입횟수타겟[타입횟수타겟['type']==0]
교환창횟수타겟 = 교환창횟수타겟.drop('type', axis=1)           
교환창횟수타겟.columns=['acc_id', '교환창타겟']

trade_ = pd.merge(개인상점횟수소스, 교환창횟수소스, on=['acc_id'], how='outer')
trade_ = pd.merge(trade_, 개인상점횟수타겟, on=['acc_id'], how='outer')
trade_ = pd.merge(trade_, 교환창횟수타겟, on=['acc_id'], how='outer')
trade_.fillna(0, inplace=True)


# In[18]:


# 일별 거래 횟수 (소스/타겟)
일별소스거래횟수 = trade.pivot_table(index=['source_acc_id'], values='type' , aggfunc='count').reset_index()
일별소스거래횟수.columns = ['acc_id', '소스거래횟수']
일별타겟거래횟수 = trade.pivot_table(index=['target_acc_id'], values='type' , aggfunc='count').reset_index()
일별타겟거래횟수.columns = ['acc_id', '타겟거래횟수']
trade_ = pd.merge(trade_, 일별소스거래횟수, on=['acc_id'], how='outer')
trade_ = pd.merge(trade_, 일별타겟거래횟수, on=['acc_id'], how='outer')
trade_.fillna(0, inplace=True)


# In[19]:


# 아이템별
소스아이템별 = trade.pivot_table(index=['source_acc_id','item_type'], values='day', aggfunc='count').unstack(1).reset_index()
소스아이템별 = 소스아이템별.fillna(0)
소스아이템별 = pd.DataFrame(소스아이템별.values)
소스아이템별.columns = ['acc_id', 'accessory', 'adena', 'armor', 'enchant_scroll', 'etc', 'spell', 'weapon']

타겟아이템별 = trade.pivot_table(index=['target_acc_id','item_type'], values='day', aggfunc='count').unstack(1).reset_index()
타겟아이템별 = 타겟아이템별.fillna(0)
타겟아이템별 = pd.DataFrame(타겟아이템별.values)
타겟아이템별.columns = ['acc_id', 'accessory', 'adena', 'armor', 'enchant_scroll', 'etc', 'spell', 'weapon']

아이템별 = pd.merge(소스아이템별, 타겟아이템별, on='acc_id', how='left')
아이템별 = 아이템별.fillna(0)
아이템별['accessory'] = 아이템별.iloc[:,1] + 아이템별.iloc[:,8]
아이템별['adena'] = 아이템별.iloc[:,2] + 아이템별.iloc[:,9]
아이템별['armor'] = 아이템별.iloc[:,3] + 아이템별.iloc[:,10]
아이템별['enchant_scroll'] = 아이템별.iloc[:,4] + 아이템별.iloc[:,11]
아이템별['etc'] = 아이템별.iloc[:,5] + 아이템별.iloc[:,12]
아이템별['spell'] = 아이템별.iloc[:,6] + 아이템별.iloc[:,13]
아이템별['weapon'] = 아이템별.iloc[:,7] + 아이템별.iloc[:,14]
아이템별.drop(아이템별.columns[1:15], axis=1, inplace=True)
trade_ = pd.merge(trade_, 아이템별, how='left', on='acc_id')


# In[20]:


temp = pd.merge(temp, trade_, on='acc_id', how='left')
temp.fillna(0, inplace=True)


# ### 1-4. combat

# In[21]:


acitivty_play = activity_og[['day', 'acc_id', 'char_id', 'playtime']]
combat_play = pd.merge(combat, acitivty_play, on=['day', 'acc_id', 'char_id'], how='left')


# In[22]:


# 제일 많이 플레이한 캐릭터 직업 
캐릭터직업 = {'acc_id':[], 'class':[]}
mp_class = combat_play.pivot_table(index=['acc_id', 'class'], values='playtime', aggfunc='sum').reset_index()
ids = mp_class['acc_id'].unique()
for acc_id in ids:
    tmp = mp_class[mp_class['acc_id']==acc_id][mp_class[mp_class['acc_id']==acc_id]['playtime'] == mp_class[mp_class['acc_id']==acc_id]['playtime'].max()].reset_index()
    캐릭터직업['acc_id'].append(acc_id)
    캐릭터직업['class'].append(tmp['class'][0])

캐릭터직업df = pd.DataFrame(캐릭터직업)
temp = pd.merge(temp, 캐릭터직업df, on='acc_id', how='left')


# In[23]:


# 제일 많이 플레이한 캐릭터 레벨
캐릭터레벨 = {'acc_id':[], 'level':[]}
mp_level = combat_play.pivot_table(index=['acc_id', 'level'], values='playtime', aggfunc='sum').reset_index()
ids = mp_level['acc_id'].unique()
for acc_id in ids:
    tmp = mp_level[mp_level['acc_id']==acc_id][mp_level[mp_level['acc_id']==acc_id]['playtime'] == mp_level[mp_level['acc_id']==acc_id]['playtime'].max()].reset_index()
    캐릭터레벨['acc_id'].append(acc_id)
    캐릭터레벨['level'].append(tmp['level'][0])

캐릭터레벨df = pd.DataFrame(캐릭터레벨)
temp = pd.merge(temp, 캐릭터레벨df, on='acc_id', how='left')


# In[24]:


combat_sum = combat.pivot_table(index=['acc_id'], aggfunc='sum').reset_index()
combat_sum = combat_sum.drop(['char_id', 'class', 'day', 'level'], axis=1)
combat_sum.fillna(0, inplace=True)


# In[25]:


combat_sum_columns = ['acc_id']
for col in combat_sum.columns[1:]:
    combat_sum_columns.append(col+'_combat_sum')
combat_sum.columns = combat_sum_columns


# ### 1-5. pledge

# In[26]:


acitivty_play = activity_og[['day', 'acc_id', 'char_id', 'playtime']]
pledge_play = pd.merge(pledge, acitivty_play, on=['day', 'acc_id', 'char_id'], how='left')


# In[27]:


# 가장 많이 플레이한 혈맹 아이디
혈맹아이디 = {'acc_id':[], 'pledge_id':[]}
mp_pledge = pledge_play.pivot_table(index=['acc_id', 'pledge_id'], values='playtime', aggfunc='sum').reset_index()
ids = mp_pledge['acc_id'].unique()
for acc_id in ids:
    tmp = mp_pledge[mp_pledge['acc_id']==acc_id][mp_pledge[mp_pledge['acc_id']==acc_id]['playtime'] == mp_pledge[mp_pledge['acc_id']==acc_id]['playtime'].max()].reset_index()
    혈맹아이디['acc_id'].append(acc_id)
    혈맹아이디['pledge_id'].append(tmp['pledge_id'][0]) 

혈맹아이디df = pd.DataFrame(혈맹아이디)


# In[28]:


혈맹원수 = pledge.pivot_table(index='pledge_id', values='acc_id', aggfunc=pd.Series.nunique).reset_index()
혈맹원수.columns = ['pledge_id', '혈맹원수']
혈맹원수 = pd.merge(혈맹아이디df, 혈맹원수, on='pledge_id', how='left')
혈맹원수.drop('pledge_id', axis=1)
temp = pd.merge(temp, 혈맹원수, on='acc_id', how='left')


# In[29]:


pledge_sum = pledge.pivot_table(index=['acc_id'], aggfunc='sum').reset_index()
pledge_sum = pledge_sum.drop(['char_id', 'day', 'pledge_id'], axis=1)
pledge_sum.fillna(0, inplace=True)


# In[30]:


pledge_sum_columns = ['acc_id']
for col in pledge_sum.columns[1:]:
    pledge_sum_columns.append(col+'_pledge_sum')
pledge_sum.columns = pledge_sum_columns


# In[31]:


temp.drop('pledge_id', axis=1, inplace=True)


# ### 1-6. merge & clustering & label

# In[32]:


data = pd.merge(activity_day_sum, payment_day_sum, on=['day', 'acc_id'], how='left')


# In[33]:


data.fillna(0, inplace=True)


# In[34]:


def timeseries(activity_df, df_before_flatten):
    activity1 = activity['acc_id'].unique()
    activity1 = pd.DataFrame(activity1)
    activity1['day'] = 1    # day 컬럼 생성

    for i in range(2,29):
        activity2 = activity['acc_id'].unique()
        activity2 = pd.DataFrame(activity2)        
        activity2['day'] = i
        activity1 = pd.concat([activity1, activity2])
       
    activity1.columns = ['acc_id', 'day']

    data = pd.merge(activity1, df_before_flatten, on=['day','acc_id'], how='left')

    return data


# In[35]:


data = timeseries(activity_og, data)
data = data.sort_values(by=['acc_id', 'day']).reset_index(drop=True)
data.fillna(0, inplace=True)


# In[36]:


def to_flat(df):
    cc = df.groupby(['acc_id']).cumcount() + 1
    flat_df = df.set_index(['acc_id', cc]).unstack().sort_index(1, level=1)
    flat_df.columns = ['_'.join(map(str,i)) for i in flat_df.columns]
    flat_df.reset_index(inplace=True)
    return flat_df


# In[37]:


data = to_flat(data)


# In[38]:


def drop_days(df):
    columns = []
    for i in range(1,29):
        columns.append('day_' + str(i))
    df.drop(columns, axis = 1, inplace = True)
    return df


# In[39]:


data = drop_days(data)


# In[40]:


data = pd.merge(data, combat_sum, on=['acc_id'], how='left')
data = pd.merge(data, pledge_sum, on=['acc_id'], how='left')
temp = pd.merge(temp, payment_day_std, on='acc_id', how='left')


# In[41]:


data = pd.merge(data, temp, on='acc_id', how='left')
data.fillna(0, inplace=True)


# In[42]:


cols = ['acc_id']
for i in range(1,29):
    cols.append('playtime_sum_'+str(i))

data_pattern_day = data[cols]
data_pattern_day_values = data_pattern_day.iloc[:,1:].values


# In[43]:


for col in data_pattern_day.columns[1:]:
    data_pattern_day[col] = data_pattern_day[col].apply(lambda x : 0 if x==0 else 1)


# In[44]:


pattern_play_list = data_pattern_day.iloc[:,1:].values.tolist()


# In[45]:


최대연속 = []
for i in range(len(pattern_play_list)):
    temp = 0
    최대연속접속일수 = 0
    for j in range(0,28):
        if pattern_play_list[i][j]==1:
            temp += 1
            if temp >= 최대연속접속일수:
                최대연속접속일수 = temp
        else:
            if temp >= 최대연속접속일수:
                최대연속접속일수 = temp
                temp = 0
                
    최대연속.append(최대연속접속일수)


# In[46]:


최대연속미접속 = []
for i in range(len(pattern_play_list)):
    temp = 0
    최대연속미접속일수 = 0
    for j in range(0,28):
        if pattern_play_list[i][j]==0:
            temp += 1
            if temp >= 최대연속미접속일수:
                최대연속미접속일수 = temp
        else:
            if temp >= 최대연속미접속일수:
                최대연속미접속일수 = temp
                temp = 0
                
    최대연속미접속.append(최대연속미접속일수)


# In[47]:


최대연속 = pd.DataFrame(최대연속)
최대연속.columns=['최대연속접속일수']
최대미연속 = pd.DataFrame(최대연속미접속)
최대미연속.columns=['최대연속미접속일수']


# In[48]:


data = pd.concat([data, 최대연속], axis=1)
data = pd.concat([data, 최대미연속], axis=1)


# In[49]:


model = KMeans(n_clusters=4, algorithm='auto', random_state=99999)
model.fit(data_pattern_day)
predict = model.predict(data_pattern_day)
data['day_playpattern'] = predict


# In[53]:


data = pd.get_dummies(data, columns=['class', 'level', 'day_playpattern'])


# In[54]:


data = pd.merge(data, label, on='acc_id', how='left')


# In[55]:


df = data.drop(['random_attacker_cnt_pledge_sum', 'random_defender_cnt_pledge_sum','combat_play_time_pledge_sum' ,
            'non_combat_play_time_pledge_sum', 'random_attacker_cnt_combat_sum', 'random_defender_cnt_combat_sum']
           , axis=1)


# In[56]:


df.to_csv('train_preprocess_1.csv', index=False)


# In[57]:


data.to_csv('train_preprocess_2.csv', index=False)


# # 2. TEST 1

# In[58]:


del data
del temp


# In[59]:


activity = pd.read_csv('../raw/test1_activity.csv')
payment = pd.read_csv('../raw/test1_payment.csv')
trade = pd.read_csv('../raw/test1_trade.csv')
pledge = pd.read_csv('../raw/test1_pledge.csv')
combat = pd.read_csv('../raw/test1_combat.csv')


# ### 2-1. activity

# In[60]:


activity_og = activity.copy()


# In[61]:


temp = activity.pivot_table(index='acc_id', values='day', aggfunc='min').reset_index()
temp['week'] = temp['day'].apply(week)
temp.columns = ['acc_id', '최초접속일', '최초접속주차']

activity['week'] = activity['day'].apply(week)
activity_week1 = activity[activity['week']==1]
activity_week2 = activity[activity['week']==2]
activity_week3 = activity[activity['week']==3]
activity_week4 = activity[activity['week']==4]

첫주차접속일 = activity_week1.pivot_table(index='acc_id', values='day', aggfunc=pd.Series.nunique).reset_index()
첫주차접속일.columns = ['acc_id', '첫주차접속일']
사주차접속일 = activity_week4.pivot_table(index='acc_id', values='day', aggfunc=pd.Series.nunique).reset_index()
사주차접속일.columns = ['acc_id', '사주차접속일']

temp = pd.merge(temp, 첫주차접속일, on='acc_id', how='left')
temp = pd.merge(temp, 사주차접속일, on='acc_id', how='left')
temp.fillna(0, inplace=True)

# unique count - 캐릭터별 최고 접속일
캐릭터별총접속일 = activity.pivot_table(index='acc_id', values='day', aggfunc='count').reset_index()
캐릭터별총접속일.columns = ['acc_id', '캐릭터별총접속일']
temp = pd.merge(temp, 캐릭터별총접속일, on='acc_id', how='left')

# unique count - 캐릭터별 최고 접속일
ID별총접속일 = activity.pivot_table(index='acc_id', values='day', aggfunc=pd.Series.nunique).reset_index()
ID별총접속일.columns = ['acc_id', 'ID별총접속일']
temp = pd.merge(temp, ID별총접속일, on='acc_id', how='left')

temp['최초접속이후비접속횟수'] = 28 - temp['최초접속일'] + 1 - temp['ID별총접속일']

캐릭터별접속일 = activity.pivot_table(index=['acc_id', 'char_id'], values='day', aggfunc=pd.Series.nunique).reset_index()
접속_28일 = 캐릭터별접속일[캐릭터별접속일['day']==28]
접속_28일캐릭터수 = 접속_28일.pivot_table(index='acc_id', values='char_id', aggfunc='count').reset_index()
접속_28일캐릭터수.columns = ['acc_id', '접속28일캐릭터수']
temp = pd.merge(temp, 접속_28일캐릭터수, on='acc_id', how='left')
temp.fillna(0, inplace=True)


# In[62]:


첫주플레이시간평균 = activity_week1.pivot_table(index='acc_id', values='playtime', aggfunc='mean').reset_index()
첫주플레이시간평균.columns = ['acc_id', '첫주플레이시간평균']
사주플레이시간평균 = activity_week4.pivot_table(index='acc_id', values='playtime', aggfunc='mean').reset_index()
사주플레이시간평균.columns = ['acc_id', '사주플레이시간평균']

temp = pd.merge(temp, 첫주플레이시간평균, on='acc_id', how='left')
temp = pd.merge(temp, 사주플레이시간평균, on='acc_id', how='left')
temp.fillna(0, inplace=True)


# In[63]:


activity = activity.drop(['char_id', 'server', 'week'], axis=1)


# In[64]:


activity_day_sum = activity.pivot_table(index=['day','acc_id'], aggfunc='sum').reset_index()
activity_day_sum.fillna(0,inplace=True)


# In[65]:


activity_day_sum_columns = ['day', 'acc_id']
for col in activity_day_sum.columns[2:]:
    activity_day_sum_columns.append(col+'_sum')
activity_day_sum.columns = activity_day_sum_columns


# ### 2-2. payment

# In[66]:


payment_day_sum = payment.pivot_table(index=['day','acc_id'], aggfunc='sum').reset_index()
payment_day_sum.fillna(0, inplace=True)


# In[67]:


payment_day_sum_columns = ['day', 'acc_id']
for col in payment_day_sum.columns[2:]:
    payment_day_sum_columns.append(col+'_sum')
payment_day_sum.columns = payment_day_sum_columns


# In[68]:


payment_day_std = payment.pivot_table(index=['acc_id'], aggfunc='std').reset_index()
payment_day_std.columns = ['acc_id', 'amount_spent_std', 'payment_day_std']


# In[69]:


최고지불금액 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc=max).reset_index()
최고지불금액.columns = ['acc_id', '최고지불금액']
temp = pd.merge(temp, 최고지불금액, on='acc_id', how='left')


# In[70]:


지불금액총합 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc=sum).reset_index()
지불금액총합.columns = ['acc_id', '지불금액총합']
temp = pd.merge(temp, 지불금액총합, on='acc_id', how='left')


# In[71]:


지불횟수 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc=pd.Series.nunique).reset_index()
지불횟수.columns = ['acc_id', '지불횟수']
temp = pd.merge(temp, 지불횟수, on='acc_id', how='left')


# In[72]:


평균지불금액 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc='mean').reset_index()
평균지불금액.columns = ['acc_id', '총평균지불금액']
temp = pd.merge(temp, 평균지불금액, on='acc_id', how='left')


# ### 2-3. trade

# In[73]:


# 일별 개인상점 횟수 및 교환창 횟수 (소스/타겟)
타입횟수소스 = trade.pivot_table(index=['source_acc_id', 'type'], values='item_type', aggfunc='count').reset_index()
개인상점횟수소스 = 타입횟수소스[타입횟수소스['type']==1]
개인상점횟수소스 = 개인상점횟수소스.drop('type', axis=1)           
개인상점횟수소스.columns=['acc_id', '개인상점소스']
교환창횟수소스 = 타입횟수소스[타입횟수소스['type']==0]
교환창횟수소스 = 교환창횟수소스.drop('type', axis=1)           
교환창횟수소스.columns=['acc_id', '교환창소스']

타입횟수타겟 = trade.pivot_table(index=['target_acc_id', 'type'], values='item_type', aggfunc='count').reset_index()
개인상점횟수타겟 = 타입횟수타겟[타입횟수타겟['type']==1]
개인상점횟수타겟 = 개인상점횟수타겟.drop('type', axis=1)           
개인상점횟수타겟.columns=['acc_id', '개인상점타겟']
교환창횟수타겟 = 타입횟수타겟[타입횟수타겟['type']==0]
교환창횟수타겟 = 교환창횟수타겟.drop('type', axis=1)           
교환창횟수타겟.columns=['acc_id', '교환창타겟']

trade_ = pd.merge(개인상점횟수소스, 교환창횟수소스, on=['acc_id'], how='outer')
trade_ = pd.merge(trade_, 개인상점횟수타겟, on=['acc_id'], how='outer')
trade_ = pd.merge(trade_, 교환창횟수타겟, on=['acc_id'], how='outer')
trade_.fillna(0, inplace=True)


# In[74]:


# 일별 거래 횟수 (소스/타겟)
일별소스거래횟수 = trade.pivot_table(index=['source_acc_id'], values='type' , aggfunc='count').reset_index()
일별소스거래횟수.columns = ['acc_id', '소스거래횟수']
일별타겟거래횟수 = trade.pivot_table(index=['target_acc_id'], values='type' , aggfunc='count').reset_index()
일별타겟거래횟수.columns = ['acc_id', '타겟거래횟수']
trade_ = pd.merge(trade_, 일별소스거래횟수, on=['acc_id'], how='outer')
trade_ = pd.merge(trade_, 일별타겟거래횟수, on=['acc_id'], how='outer')
trade_.fillna(0, inplace=True)


# In[75]:


# 아이템별
소스아이템별 = trade.pivot_table(index=['source_acc_id','item_type'], values='day', aggfunc='count').unstack(1).reset_index()
소스아이템별 = 소스아이템별.fillna(0)
소스아이템별 = pd.DataFrame(소스아이템별.values)
소스아이템별.columns = ['acc_id', 'accessory', 'adena', 'armor', 'enchant_scroll', 'etc', 'spell', 'weapon']

타겟아이템별 = trade.pivot_table(index=['target_acc_id','item_type'], values='day', aggfunc='count').unstack(1).reset_index()
타겟아이템별 = 타겟아이템별.fillna(0)
타겟아이템별 = pd.DataFrame(타겟아이템별.values)
타겟아이템별.columns = ['acc_id', 'accessory', 'adena', 'armor', 'enchant_scroll', 'etc', 'spell', 'weapon']

아이템별 = pd.merge(소스아이템별, 타겟아이템별, on='acc_id', how='left')
아이템별 = 아이템별.fillna(0)
아이템별['accessory'] = 아이템별.iloc[:,1] + 아이템별.iloc[:,8]
아이템별['adena'] = 아이템별.iloc[:,2] + 아이템별.iloc[:,9]
아이템별['armor'] = 아이템별.iloc[:,3] + 아이템별.iloc[:,10]
아이템별['enchant_scroll'] = 아이템별.iloc[:,4] + 아이템별.iloc[:,11]
아이템별['etc'] = 아이템별.iloc[:,5] + 아이템별.iloc[:,12]
아이템별['spell'] = 아이템별.iloc[:,6] + 아이템별.iloc[:,13]
아이템별['weapon'] = 아이템별.iloc[:,7] + 아이템별.iloc[:,14]
아이템별.drop(아이템별.columns[1:15], axis=1, inplace=True)
trade_ = pd.merge(trade_, 아이템별, how='left', on='acc_id')


# In[76]:


temp = pd.merge(temp, trade_, on='acc_id', how='left')
temp.fillna(0, inplace=True)


# ### 2-4. combat

# In[77]:


acitivty_play = activity_og[['day', 'acc_id', 'char_id', 'playtime']]
combat_play = pd.merge(combat, acitivty_play, on=['day', 'acc_id', 'char_id'], how='left')


# In[78]:


# 제일 많이 플레이한 캐릭터 직업 
캐릭터직업 = {'acc_id':[], 'class':[]}
mp_class = combat_play.pivot_table(index=['acc_id', 'class'], values='playtime', aggfunc='sum').reset_index()
ids = mp_class['acc_id'].unique()
for acc_id in ids:
    tmp = mp_class[mp_class['acc_id']==acc_id][mp_class[mp_class['acc_id']==acc_id]['playtime'] == mp_class[mp_class['acc_id']==acc_id]['playtime'].max()].reset_index()
    캐릭터직업['acc_id'].append(acc_id)
    캐릭터직업['class'].append(tmp['class'][0])

캐릭터직업df = pd.DataFrame(캐릭터직업)
temp = pd.merge(temp, 캐릭터직업df, on='acc_id', how='left')


# In[79]:


# 제일 많이 플레이한 캐릭터 레벨
캐릭터레벨 = {'acc_id':[], 'level':[]}
mp_level = combat_play.pivot_table(index=['acc_id', 'level'], values='playtime', aggfunc='sum').reset_index()
ids = mp_level['acc_id'].unique()
for acc_id in ids:
    tmp = mp_level[mp_level['acc_id']==acc_id][mp_level[mp_level['acc_id']==acc_id]['playtime'] == mp_level[mp_level['acc_id']==acc_id]['playtime'].max()].reset_index()
    캐릭터레벨['acc_id'].append(acc_id)
    캐릭터레벨['level'].append(tmp['level'][0])

캐릭터레벨df = pd.DataFrame(캐릭터레벨)
temp = pd.merge(temp, 캐릭터레벨df, on='acc_id', how='left')


# In[80]:


combat_sum = combat.pivot_table(index=['acc_id'], aggfunc='sum').reset_index()
combat_sum = combat_sum.drop(['char_id', 'class', 'day', 'level'], axis=1)
combat_sum.fillna(0, inplace=True)


# In[81]:


combat_sum_columns = ['acc_id']
for col in combat_sum.columns[1:]:
    combat_sum_columns.append(col+'_combat_sum')
combat_sum.columns = combat_sum_columns


# ### 2-5. pledge

# In[82]:


acitivty_play = activity_og[['day', 'acc_id', 'char_id', 'playtime']]
pledge_play = pd.merge(pledge, acitivty_play, on=['day', 'acc_id', 'char_id'], how='left')


# In[83]:


# 가장 많이 플레이한 혈맹 아이디
혈맹아이디 = {'acc_id':[], 'pledge_id':[]}
mp_pledge = pledge_play.pivot_table(index=['acc_id', 'pledge_id'], values='playtime', aggfunc='sum').reset_index()
ids = mp_pledge['acc_id'].unique()
for acc_id in ids:
    tmp = mp_pledge[mp_pledge['acc_id']==acc_id][mp_pledge[mp_pledge['acc_id']==acc_id]['playtime'] == mp_pledge[mp_pledge['acc_id']==acc_id]['playtime'].max()].reset_index()
    혈맹아이디['acc_id'].append(acc_id)
    혈맹아이디['pledge_id'].append(tmp['pledge_id'][0]) 

혈맹아이디df = pd.DataFrame(혈맹아이디)


# In[84]:


혈맹원수 = pledge.pivot_table(index='pledge_id', values='acc_id', aggfunc=pd.Series.nunique).reset_index()
혈맹원수.columns = ['pledge_id', '혈맹원수']
혈맹원수 = pd.merge(혈맹아이디df, 혈맹원수, on='pledge_id', how='left')
혈맹원수.drop('pledge_id', axis=1)
temp = pd.merge(temp, 혈맹원수, on='acc_id', how='left')


# In[85]:


pledge_sum = pledge.pivot_table(index=['acc_id'], aggfunc='sum').reset_index()
pledge_sum = pledge_sum.drop(['char_id', 'day', 'pledge_id'], axis=1)
pledge_sum.fillna(0, inplace=True)


# In[86]:


pledge_sum_columns = ['acc_id']
for col in pledge_sum.columns[1:]:
    pledge_sum_columns.append(col+'_pledge_sum')
pledge_sum.columns = pledge_sum_columns


# In[87]:


temp.drop('pledge_id', axis=1, inplace=True)


# ### 2-6. merge & clustering

# In[88]:


data = pd.merge(activity_day_sum, payment_day_sum, on=['day', 'acc_id'], how='left')


# In[89]:


data.fillna(0, inplace=True)


# In[90]:


data = timeseries(activity_og, data)
data = data.sort_values(by=['acc_id', 'day']).reset_index(drop=True)
data.fillna(0, inplace=True)


# In[91]:


data = to_flat(data)


# In[92]:


data = drop_days(data)


# In[93]:


data = pd.merge(data, combat_sum, on=['acc_id'], how='left')
data = pd.merge(data, pledge_sum, on=['acc_id'], how='left')
temp = pd.merge(temp, payment_day_std, on='acc_id', how='left')


# In[94]:


data = pd.merge(data, temp, on='acc_id', how='left')
data.fillna(0, inplace=True)


# In[95]:


cols = ['acc_id']
for i in range(1,29):
    cols.append('playtime_sum_'+str(i))


# In[96]:


data_pattern_day = data[cols]
data_pattern_day_values = data_pattern_day.iloc[:,1:].values


# In[97]:


for col in data_pattern_day.columns[1:]:
    data_pattern_day[col] = data_pattern_day[col].apply(lambda x : 0 if x==0 else 1)


# In[98]:


pattern_play_list = data_pattern_day.iloc[:,1:].values.tolist()


# In[99]:


최대연속 = []
for i in range(len(pattern_play_list)):
    temp = 0
    최대연속접속일수 = 0
    for j in range(0,28):
        if pattern_play_list[i][j]==1:
            temp += 1
            if temp >= 최대연속접속일수:
                최대연속접속일수 = temp
        else:
            if temp >= 최대연속접속일수:
                최대연속접속일수 = temp
                temp = 0
                
    최대연속.append(최대연속접속일수)


# In[100]:


최대연속미접속 = []
for i in range(len(pattern_play_list)):
    temp = 0
    최대연속미접속일수 = 0
    for j in range(0,28):
        if pattern_play_list[i][j]==0:
            temp += 1
            if temp >= 최대연속미접속일수:
                최대연속미접속일수 = temp
        else:
            if temp >= 최대연속미접속일수:
                최대연속미접속일수 = temp
                temp = 0
                
    최대연속미접속.append(최대연속미접속일수)


# In[101]:


최대연속 = pd.DataFrame(최대연속)
최대연속.columns=['최대연속접속일수']
최대미연속 = pd.DataFrame(최대연속미접속)
최대미연속.columns=['최대연속미접속일수']


# In[102]:


data = pd.concat([data, 최대연속], axis=1)
data = pd.concat([data, 최대미연속], axis=1)


# In[103]:


model = KMeans(n_clusters=4, algorithm='auto', random_state=43)
model.fit(data_pattern_day)
predict = model.predict(data_pattern_day)
data['day_playpattern'] = predict


# In[106]:


data = pd.get_dummies(data, columns=['class', 'level', 'day_playpattern'])


# In[107]:


df = data.drop(['random_attacker_cnt_pledge_sum', 'random_defender_cnt_pledge_sum','combat_play_time_pledge_sum' ,
            'non_combat_play_time_pledge_sum', 'random_attacker_cnt_combat_sum', 'random_defender_cnt_combat_sum']
           , axis=1)


# In[108]:


df.to_csv('test1_preprocess_1.csv', index=False)


# In[109]:


data.to_csv('test1_preprocess_2.csv', index=False)


# # 3. Test 2

# In[110]:


del data
del temp


# In[111]:


activity = pd.read_csv('../raw/test2_activity.csv')
payment = pd.read_csv('../raw/test2_payment.csv')
trade = pd.read_csv('../raw/test2_trade.csv')
pledge = pd.read_csv('../raw/test2_pledge.csv')
combat = pd.read_csv('../raw/test2_combat.csv')


# ### 3-1. activity

# In[112]:


activity_og = activity.copy()


# In[113]:


temp = activity.pivot_table(index='acc_id', values='day', aggfunc='min').reset_index()
temp['week'] = temp['day'].apply(week)
temp.columns = ['acc_id', '최초접속일', '최초접속주차']

activity['week'] = activity['day'].apply(week)
activity_week1 = activity[activity['week']==1]
activity_week2 = activity[activity['week']==2]
activity_week3 = activity[activity['week']==3]
activity_week4 = activity[activity['week']==4]

첫주차접속일 = activity_week1.pivot_table(index='acc_id', values='day', aggfunc=pd.Series.nunique).reset_index()
첫주차접속일.columns = ['acc_id', '첫주차접속일']
사주차접속일 = activity_week4.pivot_table(index='acc_id', values='day', aggfunc=pd.Series.nunique).reset_index()
사주차접속일.columns = ['acc_id', '사주차접속일']

temp = pd.merge(temp, 첫주차접속일, on='acc_id', how='left')
temp = pd.merge(temp, 사주차접속일, on='acc_id', how='left')
temp.fillna(0, inplace=True)

# unique count - 캐릭터별 최고 접속일
캐릭터별총접속일 = activity.pivot_table(index='acc_id', values='day', aggfunc='count').reset_index()
캐릭터별총접속일.columns = ['acc_id', '캐릭터별총접속일']
temp = pd.merge(temp, 캐릭터별총접속일, on='acc_id', how='left')

# unique count - 캐릭터별 최고 접속일
ID별총접속일 = activity.pivot_table(index='acc_id', values='day', aggfunc=pd.Series.nunique).reset_index()
ID별총접속일.columns = ['acc_id', 'ID별총접속일']
temp = pd.merge(temp, ID별총접속일, on='acc_id', how='left')

temp['최초접속이후비접속횟수'] = 28 - temp['최초접속일'] + 1 - temp['ID별총접속일']

캐릭터별접속일 = activity.pivot_table(index=['acc_id', 'char_id'], values='day', aggfunc=pd.Series.nunique).reset_index()
접속_28일 = 캐릭터별접속일[캐릭터별접속일['day']==28]
접속_28일캐릭터수 = 접속_28일.pivot_table(index='acc_id', values='char_id', aggfunc='count').reset_index()
접속_28일캐릭터수.columns = ['acc_id', '접속28일캐릭터수']
temp = pd.merge(temp, 접속_28일캐릭터수, on='acc_id', how='left')
temp.fillna(0, inplace=True)


# In[114]:


첫주플레이시간평균 = activity_week1.pivot_table(index='acc_id', values='playtime', aggfunc='mean').reset_index()
첫주플레이시간평균.columns = ['acc_id', '첫주플레이시간평균']
사주플레이시간평균 = activity_week4.pivot_table(index='acc_id', values='playtime', aggfunc='mean').reset_index()
사주플레이시간평균.columns = ['acc_id', '사주플레이시간평균']

temp = pd.merge(temp, 첫주플레이시간평균, on='acc_id', how='left')
temp = pd.merge(temp, 사주플레이시간평균, on='acc_id', how='left')
temp.fillna(0, inplace=True)


# In[115]:


activity = activity.drop(['char_id', 'server', 'week'], axis=1)


# In[116]:


activity_day_sum = activity.pivot_table(index=['day','acc_id'], aggfunc='sum').reset_index()
activity_day_sum.fillna(0,inplace=True)


# In[117]:


activity_day_sum_columns = ['day', 'acc_id']
for col in activity_day_sum.columns[2:]:
    activity_day_sum_columns.append(col+'_sum')
activity_day_sum.columns = activity_day_sum_columns


# ### 3-2. payment

# In[118]:


payment_day_sum = payment.pivot_table(index=['day','acc_id'], aggfunc='sum').reset_index()
payment_day_sum.fillna(0, inplace=True)


# In[119]:


payment_day_sum_columns = ['day', 'acc_id']
for col in payment_day_sum.columns[2:]:
    payment_day_sum_columns.append(col+'_sum')
payment_day_sum.columns = payment_day_sum_columns


# In[120]:


payment_day_std = payment.pivot_table(index=['acc_id'], aggfunc='std').reset_index()
payment_day_std.columns = ['acc_id', 'amount_spent_std', 'payment_day_std']


# In[121]:


최고지불금액 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc=max).reset_index()
최고지불금액.columns = ['acc_id', '최고지불금액']
temp = pd.merge(temp, 최고지불금액, on='acc_id', how='left')


# In[122]:


지불금액총합 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc=sum).reset_index()
지불금액총합.columns = ['acc_id', '지불금액총합']
temp = pd.merge(temp, 지불금액총합, on='acc_id', how='left')


# In[123]:


지불횟수 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc=pd.Series.nunique).reset_index()
지불횟수.columns = ['acc_id', '지불횟수']
temp = pd.merge(temp, 지불횟수, on='acc_id', how='left')


# In[124]:


평균지불금액 = payment.pivot_table(index='acc_id', values='amount_spent', aggfunc='mean').reset_index()
평균지불금액.columns = ['acc_id', '총평균지불금액']
temp = pd.merge(temp, 평균지불금액, on='acc_id', how='left')


# ### 3-3. trade

# In[125]:


# 일별 개인상점 횟수 및 교환창 횟수 (소스/타겟)
타입횟수소스 = trade.pivot_table(index=['source_acc_id', 'type'], values='item_type', aggfunc='count').reset_index()
개인상점횟수소스 = 타입횟수소스[타입횟수소스['type']==1]
개인상점횟수소스 = 개인상점횟수소스.drop('type', axis=1)           
개인상점횟수소스.columns=['acc_id', '개인상점소스']
교환창횟수소스 = 타입횟수소스[타입횟수소스['type']==0]
교환창횟수소스 = 교환창횟수소스.drop('type', axis=1)           
교환창횟수소스.columns=['acc_id', '교환창소스']

타입횟수타겟 = trade.pivot_table(index=['target_acc_id', 'type'], values='item_type', aggfunc='count').reset_index()
개인상점횟수타겟 = 타입횟수타겟[타입횟수타겟['type']==1]
개인상점횟수타겟 = 개인상점횟수타겟.drop('type', axis=1)           
개인상점횟수타겟.columns=['acc_id', '개인상점타겟']
교환창횟수타겟 = 타입횟수타겟[타입횟수타겟['type']==0]
교환창횟수타겟 = 교환창횟수타겟.drop('type', axis=1)           
교환창횟수타겟.columns=['acc_id', '교환창타겟']

trade_ = pd.merge(개인상점횟수소스, 교환창횟수소스, on=['acc_id'], how='outer')
trade_ = pd.merge(trade_, 개인상점횟수타겟, on=['acc_id'], how='outer')
trade_ = pd.merge(trade_, 교환창횟수타겟, on=['acc_id'], how='outer')
trade_.fillna(0, inplace=True)


# In[126]:


# 일별 거래 횟수 (소스/타겟)
일별소스거래횟수 = trade.pivot_table(index=['source_acc_id'], values='type' , aggfunc='count').reset_index()
일별소스거래횟수.columns = ['acc_id', '소스거래횟수']
일별타겟거래횟수 = trade.pivot_table(index=['target_acc_id'], values='type' , aggfunc='count').reset_index()
일별타겟거래횟수.columns = ['acc_id', '타겟거래횟수']
trade_ = pd.merge(trade_, 일별소스거래횟수, on=['acc_id'], how='outer')
trade_ = pd.merge(trade_, 일별타겟거래횟수, on=['acc_id'], how='outer')
trade_.fillna(0, inplace=True)


# In[127]:


# 아이템별
소스아이템별 = trade.pivot_table(index=['source_acc_id','item_type'], values='day', aggfunc='count').unstack(1).reset_index()
소스아이템별 = 소스아이템별.fillna(0)
소스아이템별 = pd.DataFrame(소스아이템별.values)
소스아이템별.columns = ['acc_id', 'accessory', 'adena', 'armor', 'enchant_scroll', 'etc', 'spell', 'weapon']

타겟아이템별 = trade.pivot_table(index=['target_acc_id','item_type'], values='day', aggfunc='count').unstack(1).reset_index()
타겟아이템별 = 타겟아이템별.fillna(0)
타겟아이템별 = pd.DataFrame(타겟아이템별.values)
타겟아이템별.columns = ['acc_id', 'accessory', 'adena', 'armor', 'enchant_scroll', 'etc', 'spell', 'weapon']

아이템별 = pd.merge(소스아이템별, 타겟아이템별, on='acc_id', how='left')
아이템별 = 아이템별.fillna(0)
아이템별['accessory'] = 아이템별.iloc[:,1] + 아이템별.iloc[:,8]
아이템별['adena'] = 아이템별.iloc[:,2] + 아이템별.iloc[:,9]
아이템별['armor'] = 아이템별.iloc[:,3] + 아이템별.iloc[:,10]
아이템별['enchant_scroll'] = 아이템별.iloc[:,4] + 아이템별.iloc[:,11]
아이템별['etc'] = 아이템별.iloc[:,5] + 아이템별.iloc[:,12]
아이템별['spell'] = 아이템별.iloc[:,6] + 아이템별.iloc[:,13]
아이템별['weapon'] = 아이템별.iloc[:,7] + 아이템별.iloc[:,14]
아이템별.drop(아이템별.columns[1:15], axis=1, inplace=True)
trade_ = pd.merge(trade_, 아이템별, how='left', on='acc_id')


# In[128]:


temp = pd.merge(temp, trade_, on='acc_id', how='left')
temp.fillna(0, inplace=True)


# ### 3-4. combat

# In[129]:


acitivty_play = activity_og[['day', 'acc_id', 'char_id', 'playtime']]
combat_play = pd.merge(combat, acitivty_play, on=['day', 'acc_id', 'char_id'], how='left')


# In[130]:


# 제일 많이 플레이한 캐릭터 직업 
캐릭터직업 = {'acc_id':[], 'class':[]}
mp_class = combat_play.pivot_table(index=['acc_id', 'class'], values='playtime', aggfunc='sum').reset_index()
ids = mp_class['acc_id'].unique()
for acc_id in ids:
    tmp = mp_class[mp_class['acc_id']==acc_id][mp_class[mp_class['acc_id']==acc_id]['playtime'] == mp_class[mp_class['acc_id']==acc_id]['playtime'].max()].reset_index()
    캐릭터직업['acc_id'].append(acc_id)
    캐릭터직업['class'].append(tmp['class'][0])

캐릭터직업df = pd.DataFrame(캐릭터직업)
temp = pd.merge(temp, 캐릭터직업df, on='acc_id', how='left')


# In[131]:


# 제일 많이 플레이한 캐릭터 레벨
캐릭터레벨 = {'acc_id':[], 'level':[]}
mp_level = combat_play.pivot_table(index=['acc_id', 'level'], values='playtime', aggfunc='sum').reset_index()
ids = mp_level['acc_id'].unique()
for acc_id in ids:
    tmp = mp_level[mp_level['acc_id']==acc_id][mp_level[mp_level['acc_id']==acc_id]['playtime'] == mp_level[mp_level['acc_id']==acc_id]['playtime'].max()].reset_index()
    캐릭터레벨['acc_id'].append(acc_id)
    캐릭터레벨['level'].append(tmp['level'][0])

캐릭터레벨df = pd.DataFrame(캐릭터레벨)
temp = pd.merge(temp, 캐릭터레벨df, on='acc_id', how='left')


# In[132]:


combat_sum = combat.pivot_table(index=['acc_id'], aggfunc='sum').reset_index()
combat_sum = combat_sum.drop(['char_id', 'class', 'day', 'level'], axis=1)
combat_sum.fillna(0, inplace=True)


# In[133]:


combat_sum_columns = ['acc_id']
for col in combat_sum.columns[1:]:
    combat_sum_columns.append(col+'_combat_sum')
combat_sum.columns = combat_sum_columns


# ### 3-5. pledge

# In[134]:


acitivty_play = activity_og[['day', 'acc_id', 'char_id', 'playtime']]
pledge_play = pd.merge(pledge, acitivty_play, on=['day', 'acc_id', 'char_id'], how='left')


# In[135]:


# 가장 많이 플레이한 혈맹 아이디
혈맹아이디 = {'acc_id':[], 'pledge_id':[]}
mp_pledge = pledge_play.pivot_table(index=['acc_id', 'pledge_id'], values='playtime', aggfunc='sum').reset_index()
ids = mp_pledge['acc_id'].unique()
for acc_id in ids:
    tmp = mp_pledge[mp_pledge['acc_id']==acc_id][mp_pledge[mp_pledge['acc_id']==acc_id]['playtime'] == mp_pledge[mp_pledge['acc_id']==acc_id]['playtime'].max()].reset_index()
    혈맹아이디['acc_id'].append(acc_id)
    혈맹아이디['pledge_id'].append(tmp['pledge_id'][0]) 

혈맹아이디df = pd.DataFrame(혈맹아이디)


# In[136]:


혈맹원수 = pledge.pivot_table(index='pledge_id', values='acc_id', aggfunc=pd.Series.nunique).reset_index()
혈맹원수.columns = ['pledge_id', '혈맹원수']
혈맹원수 = pd.merge(혈맹아이디df, 혈맹원수, on='pledge_id', how='left')
혈맹원수.drop('pledge_id', axis=1)
temp = pd.merge(temp, 혈맹원수, on='acc_id', how='left')


# In[137]:


pledge_sum = pledge.pivot_table(index=['acc_id'], aggfunc='sum').reset_index()
pledge_sum = pledge_sum.drop(['char_id', 'day', 'pledge_id'], axis=1)
pledge_sum.fillna(0, inplace=True)


# In[138]:


pledge_sum_columns = ['acc_id']
for col in pledge_sum.columns[1:]:
    pledge_sum_columns.append(col+'_pledge_sum')
pledge_sum.columns = pledge_sum_columns


# In[139]:


temp.drop('pledge_id', axis=1, inplace=True)


# ### 3-6. merge & clustering

# In[140]:


data = pd.merge(activity_day_sum, payment_day_sum, on=['day', 'acc_id'], how='left')


# In[141]:


data.fillna(0, inplace=True)


# In[142]:


data = timeseries(activity_og, data)
data = data.sort_values(by=['acc_id', 'day']).reset_index(drop=True)
data.fillna(0, inplace=True)


# In[143]:


data = to_flat(data)


# In[144]:


data = drop_days(data)


# In[145]:


data = pd.merge(data, combat_sum, on=['acc_id'], how='left')
data = pd.merge(data, pledge_sum, on=['acc_id'], how='left')
temp = pd.merge(temp, payment_day_std, on='acc_id', how='left')


# In[146]:


data = pd.merge(data, temp, on='acc_id', how='left')
data.fillna(0, inplace=True)


# In[147]:


cols = ['acc_id']
for i in range(1,29):
    cols.append('playtime_sum_'+str(i))


# In[148]:


data_pattern_day = data[cols]
data_pattern_day_values = data_pattern_day.iloc[:,1:].values


# In[149]:


for col in data_pattern_day.columns[1:]:
    data_pattern_day[col] = data_pattern_day[col].apply(lambda x : 0 if x==0 else 1)


# In[150]:


pattern_play_list = data_pattern_day.iloc[:,1:].values.tolist()


# In[151]:


최대연속 = []
for i in range(len(pattern_play_list)):
    temp = 0
    최대연속접속일수 = 0
    for j in range(0,28):
        if pattern_play_list[i][j]==1:
            temp += 1
            if temp >= 최대연속접속일수:
                최대연속접속일수 = temp
        else:
            if temp >= 최대연속접속일수:
                최대연속접속일수 = temp
                temp = 0
                
    최대연속.append(최대연속접속일수)


# In[152]:


최대연속미접속 = []
for i in range(len(pattern_play_list)):
    temp = 0
    최대연속미접속일수 = 0
    for j in range(0,28):
        if pattern_play_list[i][j]==0:
            temp += 1
            if temp >= 최대연속미접속일수:
                최대연속미접속일수 = temp
        else:
            if temp >= 최대연속미접속일수:
                최대연속미접속일수 = temp
                temp = 0
                
    최대연속미접속.append(최대연속미접속일수)


# In[153]:


최대연속 = pd.DataFrame(최대연속)
최대연속.columns=['최대연속접속일수']
최대미연속 = pd.DataFrame(최대연속미접속)
최대미연속.columns=['최대연속미접속일수']


# In[154]:


data = pd.concat([data, 최대연속], axis=1)
data = pd.concat([data, 최대미연속], axis=1)


# In[155]:


model = KMeans(n_clusters=4, algorithm='auto', random_state=43)
model.fit(data_pattern_day)
predict = model.predict(data_pattern_day)
data['day_playpattern'] = predict


# In[158]:


data = pd.get_dummies(data, columns=['class', 'level', 'day_playpattern'])


# In[159]:


df = data.drop(['random_attacker_cnt_pledge_sum', 'random_defender_cnt_pledge_sum','combat_play_time_pledge_sum' ,
            'non_combat_play_time_pledge_sum', 'random_attacker_cnt_combat_sum', 'random_defender_cnt_combat_sum']
           , axis=1)


# In[160]:


df.to_csv('test2_preprocess_1.csv', index=False)


# In[161]:


data.to_csv('test2_preprocess_2.csv', index=False)

