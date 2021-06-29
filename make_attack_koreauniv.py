#!/usr/bin/env python
# coding: utf-8

# # preprocessing
# 

# # test-make attack

# In[45]:


#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np

class AttackGeneration:
    def __init__(self, values, attack_type):
        self.seq_list = values
        self.type = attack_type
        self.attack_result = self.apply_attack()

    def apply_attack(self):
        if self.type == 1:
            at_output = self.attack_1(self.seq_list)
        elif self.type == 2:
            at_output = self.attack_2(self.seq_list)
        elif self.type == 3:
            at_output = self.attack_3(self.seq_list)
        elif self.type == 4:
            at_output = self.attack_4(self.seq_list)
        elif self.type == 5:
            at_output = self.attack_5(self.seq_list)
        elif self.type == 6:
            at_output = self.attack_6(self.seq_list)
        else:
            at_output = "Clarify the attack type..."

        return at_output

    def attack_1(self,seq_list):
#        alpha = float(np.random.uniform(0.1, 0.8))
        alpha = np.random.uniform(0.1, 0.8)
        seq_list = np.asarray(seq_list)
        return [i * alpha for i in seq_list] # , alpha


    def attack_2(self,seq_list, minOffTime=8):
        len_seq = len(seq_list)

        # minOffTime = 8
        start_time = np.random.randint(0, (len_seq - 1 - minOffTime))
        duration = np.random.randint(minOffTime, len_seq)
        end_time = min(start_time + duration, len_seq - 1)

        output = list()
        for idx, value in enumerate(seq_list):
            if (idx > start_time) & (idx < end_time):
                output.append(0)
            else:
                output.append(value)

        return output #, start_time, duration

    def attack_3(self,seq_list):
                
        gamma_list = np.random.uniform(0.1, 0.8, len(seq_list))
        seq_list = np.asarray(seq_list)

        return [x_t * gamma_t for x_t, gamma_t in zip(seq_list, gamma_list)]#, gamma_list

    def attack_4(self,seq_list):
        mean_x = np.mean(np.array(seq_list))
        gamma_list = np.random.uniform(0.1, 0.8, len(seq_list))
        return [mean_x * gamma_t for gamma_t in gamma_list]#, mean_x, gamma_list

    def attack_5(self,seq_list):
        mean_x = np.mean(np.array(seq_list))
        return [mean_x for i in seq_list]

    def attack_6(self,seq_list):
        return seq_list[::-1]

# seq = [1,2,3,4]*12
# output  = AttackGeneration(values=seq,attack_type=7).attack_result




# In[46]:



# In[38]:


import pandas as pd
import numpy as np

#data1 = pd.read_csv("C:/Users/KIM/Desktop/DSBA/개인연구/교내_스마트그리드/창의관_Dataset_RF_test.csv")
data1 =  pd.read_csv("C:/Users/KIM/Desktop/DSBA/개인연구/교내_스마트그리드/하나과학관_Dataset_RF_test.csv")
#data = np.loadtxt(open('C:/Users/KIM/Desktop/final_splined_df.csv', 'r'), delimiter=',')
#data = pd.concat([data1, data2])
data = data1
#datetime 열 삭제
data = data.loc[:,['UsedPower']]


# In[47]:


data


# In[48]:


#attack : 1%
per_attack_no = int(len(data)*0.01/ 6)
per_attack_no


# In[49]:


"""
#각각의 attack index 만들어주기
import random
blank_list=[]
for i in range(5185): #5185??
    blank_list.append(i)

#attack1_idx
attack1_idx = random.sample(blank_list,per_attack_no )

#attack2_idx
no_attack1_list =[]
for i in range(len(blank_list)):
    if blank_list[i] not in attack1_idx:
        no_attack1_list.append(i)
attack2_idx = random.sample(no_attack1_list,per_attack_no)

#attack3_idx
no_attack1_2_list=[]
for i in range(len(blank_list)):
    if blank_list[i] not in attack1_idx:
        if blank_list[i] not in attack2_idx:
            no_attack1_2_list.append(i)
        else:
            continue
    else:
        continue
attack3_idx = random.sample(no_attack1_2_list,per_attack_no)

#attack3_idx
no_attack1_2_list=[]
for i in range(len(blank_list)):
    if blank_list[i] not in attack1_idx:
        if blank_list[i] not in attack2_idx:
            no_attack1_2_list.append(i)
        else:
            continue
    else:
        continue
attack3_idx = random.sample(no_attack1_2_list,per_attack_no)

#attack4_idx
no_attack1_2_3_list=[]
for i in range(len(blank_list)):
    if blank_list[i] not in attack1_idx:
        if blank_list[i] not in attack2_idx:
            if blank_list[i] not in attack3_idx:
                no_attack1_2_3_list.append(i)
            else:
                continue
        else:
            continue
    else:
        continue
attack4_idx = random.sample(no_attack1_2_3_list,per_attack_no)

#attack5_idx
no_attack1_2_3_4_list=[]
for i in range(len(blank_list)):
    if blank_list[i] not in attack1_idx:
        if blank_list[i] not in attack2_idx:
            if blank_list[i] not in attack3_idx:
                if blank_list[i] not in attack4_idx:
                    no_attack1_2_3_4_list.append(i)
                else:
                    continue
            else:
                continue
        else:
            continue
    else:
        continue
attack5_idx = random.sample(no_attack1_2_3_4_list,per_attack_no)

#attack6_idx
no_attack1_2_3_4_5_list=[]
for i in range(len(blank_list)):
    if blank_list[i] not in attack1_idx:
        if blank_list[i] not in attack2_idx:
            if blank_list[i] not in attack3_idx:
                if blank_list[i] not in attack4_idx:
                    if blank_list[i] not in attack5_idx:
                        no_attack1_2_3_4_5_list.append(i)
                    else:
                        continue
                else:
                    continue
            else:
                continue
        else:
            continue
    else:
        continue
attack6_idx = random.sample(no_attack1_2_3_4_5_list,per_attack_no)
"""


# In[50]:


data[len(data)-2*per_attack_no: len(data)-per_attack_no]


# In[51]:


#attack: per_attack_no개씩 6개 뽑기(시작 시점: 랜덤)

# generate random floating point values
from random import seed
from random import randint
# seed random number generator
seed(1)
# generate random numbers between 0-1
start_idx1 = randint(0,len(data))
#attack1_index
attack1_idx = data[start_idx1: start_idx1+per_attack_no].index.values
#attack1에 할당된 idx 제외하기
for i in attack1_idx:
    #prac=data.drop(i,inplace=True)
    wo1 = data.drop(data.index[i])
    
#generate start_idx2
list2=[]
for i in data.iloc[0:len(data)-per_attack_no].index:
    if i not in attack1_idx:
        list2.append(i)
start_idx2 =np.random.choice(list2)
#attack2_index
attack2_idx = []
for i in range(per_attack_no):
    attack2_idx.append(start_idx2+i)
attack2_idx = np.array(attack2_idx)


#generate start_idx3
list3=[]
for i in data.iloc[0:len(data)-per_attack_no].index:
    if i not in attack1_idx:
        if i not in attack2_idx:
            list3.append(i)
start_idx3 =np.random.choice(list3)
#attack3_index
attack3_idx = []
for i in range(per_attack_no):
    attack3_idx.append(start_idx3+i)
    
attack3_idx = np.array(attack3_idx)


#generate start_idx4
list4=[]
for i in data.iloc[0:len(data)-per_attack_no].index:
    if i not in attack1_idx:
        if i not in attack2_idx:
            if i not in attack3_idx:
                list4.append(i)
start_idx4 =np.random.choice(list4)
#attack4_index
attack4_idx = []
for i in range(per_attack_no):
    attack4_idx.append(start_idx4+i)
    
attack4_idx = np.array(attack4_idx)

#generate start_idx5
list5=[]
for i in data.iloc[0:len(data)-per_attack_no].index:
    if i not in attack1_idx:
        if i not in attack2_idx:
            if i not in attack3_idx:
                if i not in attack4_idx:
                    list5.append(i)
start_idx5 =np.random.choice(list5)
#attack5_index
attack5_idx = []
for i in range(per_attack_no):
    attack5_idx.append(start_idx5+i)
    
attack5_idx = np.array(attack5_idx)

#generate start_idx6
list6=[]
for i in data.iloc[0:len(data)-per_attack_no].index:
    if i not in attack1_idx:
        if i not in attack2_idx:
            if i not in attack3_idx:
                if i not in attack4_idx:
                    if i not in attack5_idx:
                        list6.append(i)
start_idx6 =np.random.choice(list6)
#attack6_index
attack6_idx = []
for i in range(per_attack_no):
    attack6_idx.append(start_idx6+i)
    
attack6_idx = np.array(attack6_idx)


# In[52]:


print('attack1의 index값', attack1_idx)
print('attack2의 index값', attack2_idx)
print('attack3의 index값', attack3_idx)
print('attack4의 index값', attack4_idx)
print('attack5의 index값', attack5_idx)
print('attack6의 index값', attack6_idx)


# In[53]:


# # attack1
only_attack1 = pd.DataFrame(index=range(0)) #빈 dataframe 생성

#그 index 만 추출: only_attack1

for i in attack1_idx:
    only_attack1 = only_attack1.append(data.iloc[i])
#only_attack1 = only_attack1.drop(only_attack1.columns[[1]], axis =1)

#함수 +  값 변형
alpha = np.random.uniform(0.1, 0.8)
for i in range(only_attack1.shape[0]):

    only_attack1.iloc[i] = alpha * only_attack1.iloc[i]

only_attack1 = pd.DataFrame(only_attack1)

#replace
for i in only_attack1.index.values:
    data.loc[i, 'UsedPower']= float(only_attack1.loc[i].iloc[0])
    data.loc[i,'label'] = 1


# In[54]:


# # attack2
only_attack2= pd.DataFrame(index=range(0))

#그 index 만 추출: only_attack2
for i in attack2_idx:
    only_attack2 = only_attack2.append(data.iloc[i])
#only_attack2 = only_attack2.drop('Unnamed: 0', axis =1)

#함수 +  값 변형

def attack_2(seq_list, minOffTime=1):
    len_seq = len(seq_list)

        # minOffTime = 8
    start_time = np.random.randint(0, (len_seq - 1 - minOffTime)) 
    duration = np.random.randint(minOffTime, len_seq)
    end_time = min(start_time + duration, len_seq - 1)

    output = list()
    #for idx, value in enumerate(seq_list):
    for idx, (_, value) in enumerate(seq_list.iterrows()):

    
        if (idx > start_time) & (idx < end_time):
            output.append(0)
        else:
            output.append(value)

    return output #, start_time, duration

#only_attack2 = attack_2(only_attack2)

#함수적용
a=  attack_2(only_attack2)

#각종 전처리
#for i in range(len(a)):
#    a[i] = a[i][0]
    
a_df= pd.DataFrame(index=range(per_attack_no))
a_df['UsedPower'] = a
a_df['idx'] = attack2_idx

a_df.set_index(a_df['idx'], inplace =True)

#replace
for i in a_df.index.tolist():
    if type(a_df.loc[i][0]) == int:
        data.loc[i,'UsedPower'] =0
        data.loc[i, 'label']=1

    else:   
        data.loc[i, 'UsedPower']= a_df.loc[i].values[0][0]


# In[55]:


data


# In[56]:


# # attack3
only_attack3 = pd.DataFrame(index=range(0))

#그 index 만 추출: only_attack1
for i in attack3_idx:
    only_attack3 = only_attack3.append(data.iloc[i])
only_attack3 = only_attack3.drop(only_attack3.columns[[1]], axis =1)

#함수 +  값 변형
def attack_3(seq_list):
                
    gamma_list = np.random.uniform(0.1, 0.8, len(seq_list))
    seq_list = np.asarray(seq_list)
    
    return [x_t * gamma_t for x_t, gamma_t in zip(seq_list, gamma_list)]#, gamma_list

#함수 적용
b= attack_3(only_attack3)
#각종 전처리
for i in range(len(b)):
    b[i] = b[i][0]
    
b_df= pd.DataFrame(index=range(per_attack_no))
b_df['UsedPower'] = b
b_df['idx'] = attack3_idx

b_df.set_index(b_df['idx'], inplace =True)

#replace
for i in b_df.index.values:
    data.loc[i, 'UsedPower']= b_df.loc[i].values[0]
    data.loc[i, 'label']=1


# In[57]:


# # attack4

only_attack4 = pd.DataFrame(index=range(0))

#그 index 만 추출: only_attack1
for i in attack4_idx:
    only_attack4 = only_attack4.append(data.iloc[i])
only_attack4 = only_attack4.drop(only_attack4.columns[[1]], axis =1)

#함수 +  값 변형
def attack_4(seq_list):
    mean_x = np.mean(np.array(seq_list))
    gamma_list = np.random.uniform(0.1, 0.8, len(seq_list))
    return [mean_x * gamma_t for gamma_t in gamma_list]#, mean_x, gamma_list

#함수 적용
c= attack_4(only_attack4)
#각종 전처리

c_df= pd.DataFrame(index=range(per_attack_no))
c_df['UsedPower'] = c
c_df['idx'] = attack4_idx

c_df.set_index(c_df['idx'], inplace =True)

#replace
for i in c_df.index.values:
    data.loc[i, 'UsedPower']= c_df.loc[i].values[0]
    data.loc[i, 'label']=1


# In[58]:


# # attack5

only_attack5 = pd.DataFrame(index=range(0))

#그 index 만 추출: only_attack1
for i in attack5_idx:
    only_attack5 = only_attack5.append(data.iloc[i])
only_attack5 = only_attack5.drop(only_attack5.columns[[1]], axis =1)

#함수 +  값 변형
def attack_5(seq_list):
    mean_x = np.mean(np.array(seq_list))
    return [mean_x for i in seq_list]

#함수 적용
d= attack_5(only_attack5)

#각종 전처리

d_df= pd.DataFrame(index=range(per_attack_no))
d_df['UsedPower'] = float(d[0])
d_df['idx'] = attack5_idx

d_df.set_index(d_df['idx'], inplace =True)


#replace
for i in d_df.index.values:
    data.loc[i, 'UsedPower']= d_df.loc[i].values[0]
    data.loc[i, 'label']=1


# In[59]:


# # attack6

only_attack6 = pd.DataFrame(index=range(0))

#그 index 만 추출: only_attack1
for i in attack6_idx:
    only_attack6 = only_attack6.append(data.iloc[i])
only_attack6 = only_attack6.drop(only_attack6.columns[[1]], axis =1)
print(only_attack6)
#함수 +  값 변형
def attack_6(seq_list):
    return seq_list[::-1]

#함수 적용
e= attack_6(only_attack6)
print(e)
#각종 전처리

e_df= pd.DataFrame(index=range(per_attack_no))
e_df['UsedPower'] = e['UsedPower']
e_df['idx'] = attack6_idx

e_df.set_index(e_df['idx'], inplace =True)


# In[60]:


for i in range(len(e_df)):
    e_df['UsedPower'].iloc[i] = e.values.tolist()[i][0]
print(e_df)


# In[61]:



#replace
for i in e_df.index.values:
    data.loc[i, 'UsedPower']= e_df.loc[i].values[0]
    data.loc[i, 'label']=1


# In[62]:


data


# In[63]:


#label NAN인거 다 0으로 
data['label'] = data['label'].fillna(0)


# In[64]:


data


# In[65]:


data.to_csv('C:/Users/KIM/Desktop/DSBA/개인연구/교내_스마트그리드/hanagwahak-koreauniv_test_with_attack.csv')


# In[ ]:




