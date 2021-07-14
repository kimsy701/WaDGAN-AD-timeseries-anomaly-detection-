##############################################################################
#install libraries and packages
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import math
import random


##############################################################################
#read data#
data = pd.read_csv('./data/dataC.csv')

#split to train, val, test 6:2:2
train = data[:int(data.shape[0]*0.6)]
val = data[int(data.shape[0]*0.6) : int(data.shape[0]* 0.8)]
test =  data[int(data.shape[0]*0.8):]

#bring only 'UsedPower' and 'label' column
train =pd.concat([train.UsedPower, train.label], axis= 1)
val =pd.concat([val.UsedPower, val.label], axis= 1)
test =pd.concat([test.UsedPower, test.label], axis= 1)

#select window size with validation set 이건 다른 코드에서

#training data to csv
train.to_csv('./data/split/dataC_train.csv')

"""
##############################################################################
#make abnormal data
#make attack to validation set and test set
##we can maek six types of attacks like this (not used in the paper)



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

    def attack_1(self, seq_list):
        #        alpha = float(np.random.uniform(0.1, 0.8))
        alpha = np.random.uniform(0.1, 0.8)
        seq_list = np.asarray(seq_list)
        return [i * alpha for i in seq_list]  # , alpha

    def attack_2(self, seq_list, minOffTime=8):
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

        return output  # , start_time, duration

    def attack_3(self, seq_list):

        gamma_list = np.random.uniform(0.1, 0.8, len(seq_list))
        seq_list = np.asarray(seq_list)

        return [x_t * gamma_t for x_t, gamma_t in zip(seq_list, gamma_list)]  # , gamma_list

    def attack_4(self, seq_list):
        mean_x = np.mean(np.array(seq_list))
        gamma_list = np.random.uniform(0.1, 0.8, len(seq_list))
        return [mean_x * gamma_t for gamma_t in gamma_list]  # , mean_x, gamma_list

    def attack_5(self, seq_list):
        mean_x = np.mean(np.array(seq_list))
        return [mean_x for i in seq_list]

    def attack_6(self, seq_list):
        return seq_list[::-1]

"""
# attack is only 1% of validation data


#empty dataframe for indexes
index_df =  pd.DataFrame(index=range(0,12), columns=['index'])

attack_no = int(len(val)*0.01) #280
attack_no_per_section = 24 #abnormal state for 6 hours
section_no = math.ceil(attack_no/attack_no_per_section) #12 #twelve sections total #4 sections per scenario

temp_val= val.reset_index(drop=True) #validation data shape
val_index = temp_val.index.values.tolist()

for _ in range(23):
    val_index.pop() #erase last 23 index to extract full 24 indexes in every circumstances

#fill indexes in index_df
for j in range(12):
    start_index =random.choice(val_index)
    index1 = [start_index+i for i in range(attack_no_per_section)]
    index_df.iloc[j] = [index1]
    #erase index1 from temp_val
    for k in index1:
        val_index.remove(k)


#make data to abnormal data, and change each label from 0 to 1

# 함수 +  값 변형
def scenario1(seq_list):
    gamma_list = np.random.uniform(6, 10, len(seq_list))
    seq_list = np.asarray(seq_list)

    return [x_t * gamma_t for x_t, gamma_t in zip(seq_list, gamma_list)]  # , gamma_list

def scenario2(seq_list):
    gamma_list = np.random.uniform(0.1,0.5, len(seq_list))
    seq_list = np.asarray(seq_list)

    return [x_t * gamma_t for x_t, gamma_t in zip(seq_list, gamma_list)]  # , gamma_list

def scenario3(seq_list):
    alpha = np.random.uniform(0.1,0.8)

    return seq_list * alpha



# #for validation set
# # attack1 for 4 times
for k in range(4):
    only_attack1 = pd.DataFrame(index=range(0))  # 빈 dataframe 생성

    # 그 index 만 추출: only_attack1
    for i in index_df.iloc[k]:
        for j in i:
            only_attack1 = only_attack1.append(temp_val.iloc[j])
    only_attack1.UsedPower = scenario1(only_attack1.UsedPower)
    only_attack1.label = 1

    #replace
    for i in index_df.iloc[k]:
        for j in i:
            temp_val.iloc[j] =only_attack1.loc[j]

# # attack2 for 4 times
for k in range(4,8):
    only_attack2 = pd.DataFrame(index=range(0))  # 빈 dataframe 생성

    # 그 index 만 추출: only_attack1
    for i in index_df.iloc[k]:
        for j in i:
            only_attack2 = only_attack2.append(temp_val.iloc[j])
    only_attack2.UsedPower = scenario2(only_attack2.UsedPower)
    only_attack2.label = 1

    #replace
    for i in index_df.iloc[k]:
        for j in i:
            temp_val.iloc[j] =only_attack2.loc[j]

# # attack3 for 4 times
for k in range(8,12):
    only_attack3 = pd.DataFrame(index=range(0))  # 빈 dataframe 생성

    # 그 index 만 추출: only_attack1
    for i in index_df.iloc[k]:
        for j in i:
            only_attack3 = only_attack3.append(temp_val.iloc[j])
    only_attack3.UsedPower = scenario3(only_attack3.UsedPower)
    only_attack3.label = 1

    #replace
    for i in index_df.iloc[k]:
        for j in i:
            temp_val.iloc[j] =only_attack3.loc[j]


#to_csv
temp_val.to_csv('./data/split/dataC_val.csv')


# #for test set
# attack is only 1% of test data


#empty dataframe for indexes
index_df =  pd.DataFrame(index=range(0,12), columns=['index'])

attack_no = int(len(test)*0.01) #280
attack_no_per_section = 24 #abnormal state for 6 hours
section_no = math.ceil(attack_no/attack_no_per_section) #12 #twelve sections total #4 sections per scenario

temp_test= test.reset_index(drop=True) #validation data shape
test_index = temp_test.index.values.tolist()

for _ in range(23):
    test_index.pop() #erase last 23 index to extract full 24 indexes in every circumstances

#fill indexes in index_df
for j in range(12):
    start_index =random.choice(test_index)
    index1 = [start_index+i for i in range(attack_no_per_section)]
    index_df.iloc[j] = [index1]
    #erase index1 from temp_val
    for k in index1:
        test_index.remove(k)



# #for testset 
# # attack1 for 4 times
for k in range(4):
    only_attack1 = pd.DataFrame(index=range(0))  # 빈 dataframe 생성

    # 그 index 만 추출: only_attack1
    for i in index_df.iloc[k]:
        for j in i:
            only_attack1 = only_attack1.append(temp_test.iloc[j])
    only_attack1.UsedPower = scenario1(only_attack1.UsedPower)
    only_attack1.label = 1

    #replace
    for i in index_df.iloc[k]:
        for j in i:
            temp_test.iloc[j] =only_attack1.loc[j]

# # attack2 for 4 times
for k in range(4,8):
    only_attack2 = pd.DataFrame(index=range(0))  # 빈 dataframe 생성

    # 그 index 만 추출: only_attack1
    for i in index_df.iloc[k]:
        for j in i:
            only_attack2 = only_attack2.append(temp_test.iloc[j])
    only_attack2.UsedPower = scenario2(only_attack2.UsedPower)
    only_attack2.label = 1

    #replace
    for i in index_df.iloc[k]:
        for j in i:
            temp_test.iloc[j] =only_attack2.loc[j]

# # attack3 for 4 times
for k in range(8,12):
    only_attack3 = pd.DataFrame(index=range(0))  # 빈 dataframe 생성

    # 그 index 만 추출: only_attack1
    for i in index_df.iloc[k]:
        for j in i:
            only_attack3 = only_attack3.append(temp_test.iloc[j])
    only_attack3.UsedPower = scenario3(only_attack3.UsedPower)
    only_attack3.label = 1

    #replace
    for i in index_df.iloc[k]:
        for j in i:
            temp_test.iloc[j] =only_attack3.loc[j]


#to_csv
temp_test.to_csv('./data/split/dataC_test.csv')





