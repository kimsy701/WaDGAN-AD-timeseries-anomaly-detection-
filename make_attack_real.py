import pandas as pd
import numpy as np

class AttackGeneration:

    def __init__(self, dataframe, attack1_idx, attack2_idx, attack3_idx, attack4_idx, attack5_idx, attack6_idx) :
        self.dataframe = dataframe
        self.attack1_idx = attack1_idx
        self.attack2_idx = attack2_idx
        self.attack3_idx = attack3_idx
        self.attack4_idx = attack4_idx
        self.attack5_idx = attack5_idx
        self.attack6_idx = attack6_idx

    def attack1(self):
        # # attack1
        only_attack1 = pd.DataFrame(index=range(0))  # 빈 dataframe 생성

        # 그 index 만 추출: only_attack1

        for i in self.attack1_idx:
            only_attack1 = only_attack1.append(self.dataframe.iloc[i])
        # only_attack1 = only_attack1.drop(only_attack1.columns[[1]], axis =1)

        # 함수 +  값 변형
        alpha = np.random.uniform(0.1, 0.8)
        for i in range(only_attack1.shape[0]):
            only_attack1.iloc[i] = alpha * only_attack1.iloc[i]

        only_attack1 = pd.DataFrame(only_attack1)

        # replace
        for i in only_attack1.index.values:
            self.dataframe.loc[i, 'UsedPower'] = float(only_attack1.loc[i].iloc[0])
            self.dataframe.loc[i, 'label'] = 1

        return self.dataframe

    def attack2(self):
        # # attack2
        only_attack2 = pd.DataFrame(index=range(0))

        # 그 index 만 추출: only_attack2
        for i in self.attack2_idx:
            only_attack2 = only_attack2.append(self.dataframe.iloc[i])

        # only_attack2 = only_attack2.drop('Unnamed: 0', axis =1)

        # 함수 +  값 변형

        def attack_2_func(seq_list, minOffTime=1):
            len_seq = len(seq_list)

            # minOffTime = 8
            start_time = np.random.randint(0, (len_seq - 1 - minOffTime))
            duration = np.random.randint(minOffTime, len_seq)
            end_time = min(start_time + duration, len_seq - 1)

            output = list()
            # for idx, value in enumerate(seq_list):
            for idx, (_, value) in enumerate(seq_list.iterrows()):

                if (idx > start_time) & (idx < end_time):
                    output.append(0)
                else:
                    output.append(value)

            return output  # , start_time, duration

        # only_attack2 = attack_2(only_attack2)

        # 함수적용
        a = attack_2_func(only_attack2)

        # 각종 전처리
        # for i in range(len(a)):
        #    a[i] = a[i][0]

        a_df = pd.DataFrame(index=range(self.dataframe.shape[0]//600))
        a_df['UsedPower'] = a
        a_df['idx'] = self.attack2_idx

        a_df.set_index(a_df['idx'], inplace=True)

        # replace
        for i in a_df.index.tolist():
            if type(a_df.loc[i][0]) == int:
                self.dataframe.loc[i, 'UsedPower'] = 0
                self.dataframe.loc[i, 'label'] = 1

            else:
                self.dataframe.loc[i, 'UsedPower'] = a_df.loc[i].values[0][0]

        return self.dataframe

    def attack3(self):
        # # attack3
        only_attack3 = pd.DataFrame(index=range(0))

        # 그 index 만 추출: only_attack1
        for i in self.attack3_idx:
            only_attack3 = only_attack3.append(self.dataframe.iloc[i])
        only_attack3 = only_attack3.drop(only_attack3.columns[[1]], axis=1)

        # 함수 +  값 변형
        def attack_3_func(seq_list):

            gamma_list = np.random.uniform(0.1, 0.8, len(seq_list))
            seq_list = np.asarray(seq_list)

            return [x_t * gamma_t for x_t, gamma_t in zip(seq_list, gamma_list)]  # , gamma_list

        # 함수 적용
        b = attack_3_func(only_attack3)
        # 각종 전처리
        for i in range(len(b)):
            b[i] = b[i][0]

        b_df = pd.DataFrame(index=range(self.dataframe.shape[0]//600))
        b_df['UsedPower'] = b
        b_df['idx'] = self.attack3_idx

        b_df.set_index(b_df['idx'], inplace=True)

        # replace
        for i in b_df.index.values:
            self.dataframe.loc[i, 'UsedPower'] = b_df.loc[i].values[0]
            self.dataframe.loc[i, 'label'] = 1

        return self.dataframe

    def attack4(self):
        # # attack4

        only_attack4 = pd.DataFrame(index=range(0))

        # 그 index 만 추출: only_attack1
        for i in self.attack4_idx:
            only_attack4 = only_attack4.append(self.dataframe.iloc[i])
        only_attack4 = only_attack4.drop(only_attack4.columns[[1]], axis=1)

        # 함수 +  값 변형
        def attack_4_func(seq_list):
            mean_x = np.mean(np.array(seq_list))
            gamma_list = np.random.uniform(0.1, 0.8, len(seq_list))
            return [mean_x * gamma_t for gamma_t in gamma_list]  # , mean_x, gamma_list

        # 함수 적용
        c = attack_4_func(only_attack4)
        # 각종 전처리

        c_df = pd.DataFrame(index=range(self.dataframe.shape[0]//600))
        c_df['UsedPower'] = c
        c_df['idx'] = self.attack4_idx

        c_df.set_index(c_df['idx'], inplace=True)

        # replace
        for i in c_df.index.values:
            self.dataframe.loc[i, 'UsedPower'] = c_df.loc[i].values[0]
            self.dataframe.loc[i, 'label'] = 1

        return self.dataframe

    def attack5(self):
        # # attack5

        only_attack5 = pd.DataFrame(index=range(0))

        # 그 index 만 추출: only_attack1
        for i in self.attack5_idx:
            only_attack5 = only_attack5.append(self.dataframe.iloc[i])
        only_attack5 = only_attack5.drop(only_attack5.columns[[1]], axis=1)

        # 함수 +  값 변형
        def attack_5_func(seq_list):
            mean_x = np.mean(np.array(seq_list))
            return [mean_x for i in seq_list]

        # 함수 적용
        d = attack_5_func(only_attack5)

        # 각종 전처리

        d_df = pd.DataFrame(index=range(self.dataframe.shape[0]//600))
        d_df['UsedPower'] = float(d[0])
        d_df['idx'] = self.attack5_idx

        d_df.set_index(d_df['idx'], inplace=True)

        # replace
        for i in d_df.index.values:
            self.dataframe.loc[i, 'UsedPower'] = d_df.loc[i].values[0]
            self.dataframe.loc[i, 'label'] = 1

        return self.dataframe


    def attack6(self):
        # # attack6

        only_attack6 = pd.DataFrame(index=range(0))

        # 그 index 만 추출: only_attack1
        for i in self.attack6_idx:
            only_attack6 = only_attack6.append(self.dataframe.iloc[i])
        only_attack6 = only_attack6.drop(only_attack6.columns[[1]], axis=1)
        print(only_attack6)

        # 함수 +  값 변형
        def attack_6_func(seq_list):
            return seq_list[::-1]

        # 함수 적용
        e = attack_6_func(only_attack6)

        # 각종 전처리

        e_df = pd.DataFrame(index=range(0))
        e_df['UsedPower'] = e['UsedPower']
        e_df['idx'] = self.attack6_idx

        e_df.set_index(e_df['idx'], inplace=True)
        e_df = e_df.drop(e_df.columns[[1]], axis=1)

        # replace
        for i in e_df.index.values:
            self.dataframe.loc[i, 'UsedPower'] = e_df.loc[i].values[0]
            self.dataframe.loc[i, 'label'] = 1


        return self.dataframe






