# -*- coding: utf-8 -*-
"""changui_G_diff_D_diff.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kUHwm-71EnwEiD0DJr3SpGpC-Npnilu0
"""

# Commented out IPython magic to ensure Python compatibility.

from nab_dataset import NabDataset
from models.recurrent_models_pyramid_4 import LSTMGenerator, LSTMDiscriminator
from make_attack_real import AttackGeneration



# Commented out IPython magic to ensure Python compatibility.
#nokji_same_hidden 으로, 따라서 training 할 때 -12
#창의관은 그냥 20
#for colab



#Import required libraries
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torch.nn.init as init
from torch.autograd import Variable
import datetime
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from random import randint

#Define basic settings for Adversarial Training
class ArgsTrn:
    workers=4
    batch_size=32
    epochs=20
    lr=0.0002
    cuda = True
    #manualSeed=2
opt_trn=ArgsTrn()

#torch.manual_seed(opt_trn.manualSeed)
#cudnn.benchmark = True

area_list=[]
for i in range(4,5):
    #####################################################################################################
    ### 데이터 불러오기 ###

    seed= 10**i
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark=True

    data = pd.read_csv('/content/drive/My Drive/하얀노트북/하얀노트북/창의관_Dataset_RF.csv')
    X_train, X_test, y_train, y_test = train_test_split(data['UsedPower'], data['label'],
                                                        test_size=0.20, random_state=seed)

    X_train.reset_index(drop = True, inplace = True)
    y_train.reset_index(drop = True,inplace = True)

    X_test.reset_index(drop = True,inplace = True)
    y_test.reset_index(drop = True,inplace = True)


    #concat and save
    train = pd.concat([X_train, y_train], axis =1)
    train.to_csv('/content/drive/My Drive/하얀노트북/하얀노트북/split with seed/changui_train.csv')


    # Nabdataset으로 window size 별로 된 데이터 구축
    # location of datasets and category
    # end_name = 'cpu_utilization_asg_misconfiguration.csv' # dataset name
    end_name = '하얀노트북/split with seed/changui_train.csv'  # ISSDA의 80%인 training data
    data_file = '하얀노트북/' + end_name  # dataset category and dataset name
    key = 'realKnownCause/' + end_name  # This key is used for reading anomaly labels


    # settings for data loader
    class DataSettings:

        def __init__(self):
            # self.BASE = 'D:\\ResearchDataGtx1060\\AnomalyDetectionData\\NabDataset\\'
            self.BASE = '/content/drive/My Drive/'
            # self.label_file = 'labels\\combined_windows.json'
            self.data_file = data_file
            self.key = key
            self.train = True


    data_settings = DataSettings()

    # define dataset object and data loader object for NAB dataset
    dataset = NabDataset(data_settings=data_settings)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt_trn.batch_size,
                                             shuffle=True, num_workers=int(opt_trn.workers))


    #test에 1%를 attack으로 바꾸기
        #test를 tensor에서 dataframe으로

    num_attack = len(y_test)//(100*6)
    #idx 만들기
    # generate random floating point values

    # seed random number generator
    # generate random numbers between 0-1
    #seed(1)
    start_idx1 = randint(0, len(y_test))
    per_attack_no = len(y_test)//600
    # attack1_index
    attack1_idx = X_test[start_idx1: start_idx1 + per_attack_no].index.values
    # attack1에 할당된 idx 제외하기
    for i in attack1_idx:
        # prac=data.drop(i,inplace=True)
        wo1 = X_test.drop(X_test.index[i])

    # generate start_idx2
    list2 = []
    for i in X_test.iloc[0:len(X_test) - per_attack_no].index:
        if i not in attack1_idx:
            list2.append(i)
    start_idx2 = np.random.choice(list2)
    # attack2_index
    attack2_idx = []
    for i in range(per_attack_no):
        attack2_idx.append(start_idx2 + i)
    attack2_idx = np.array(attack2_idx)

    # generate start_idx3
    list3 = []
    for i in X_test.iloc[0:len(X_test) - per_attack_no].index:
        if i not in attack1_idx:
            if i not in attack2_idx:
                list3.append(i)
    start_idx3 = np.random.choice(list3)
    # attack3_index
    attack3_idx = []
    for i in range(per_attack_no):
        attack3_idx.append(start_idx3 + i)

    attack3_idx = np.array(attack3_idx)

    # generate start_idx4
    list4 = []
    for i in X_test.iloc[0:len(X_test) - per_attack_no].index:
        if i not in attack1_idx:
            if i not in attack2_idx:
                if i not in attack3_idx:
                    list4.append(i)
    start_idx4 = np.random.choice(list4)
    # attack4_index
    attack4_idx = []
    for i in range(per_attack_no):
        attack4_idx.append(start_idx4 + i)

    attack4_idx = np.array(attack4_idx)

    # generate start_idx5
    list5 = []
    for i in X_test.iloc[0:len(X_test) - per_attack_no].index:
        if i not in attack1_idx:
            if i not in attack2_idx:
                if i not in attack3_idx:
                    if i not in attack4_idx:
                        list5.append(i)
    start_idx5 = np.random.choice(list5)
    # attack5_index
    attack5_idx = []
    for i in range(per_attack_no):
        attack5_idx.append(start_idx5 + i)

    attack5_idx = np.array(attack5_idx)

    # generate start_idx6
    list6 = []
    for i in X_test.iloc[0:len(X_test) - per_attack_no].index:
        if i not in attack1_idx:
            if i not in attack2_idx:
                if i not in attack3_idx:
                    if i not in attack4_idx:
                        if i not in attack5_idx:
                            list6.append(i)
    start_idx6 = np.random.choice(list6)
    # attack6_index
    attack6_idx = []
    for i in range(per_attack_no):
        attack6_idx.append(start_idx6 + i)

    attack6_idx = np.array(attack6_idx)

    # X_test 중 28개를 뽑고,각각 96개 중에 48개를 attack1으로.
    test = pd.concat([X_test, y_test], axis = 1)

    attackgeneration = AttackGeneration(test, attack1_idx,attack2_idx,attack3_idx,attack4_idx,attack5_idx,attack6_idx)

    test = attackgeneration.attack1()
    test = attackgeneration.attack2()
    test = attackgeneration.attack3()
    test = attackgeneration.attack4()
    test = attackgeneration.attack5()
    test = attackgeneration.attack6()


    #dataframe to csv
    test.to_csv('/content/drive/My Drive/하얀노트북/하얀노트북/split with seed/changui_test.csv')

    ##########################################################################################################
    ###모델 구축 ###

    #setup models
    #print(torch.cuda.device(0))
    #print(torch.cuda.device_count())
    #print(torch.cuda.get_device_name(0))
    #print(torch.cuda.is_available())

    device = torch.device("cuda:0" if opt_trn.cuda else "cpu") # select the device
    #device = torch.device("cpu")
    seq_len = dataset.window_length # sequence length is equal to the window length
    in_dim = dataset.n_feature # input dimension is same as number of feature

    # Create generator and discriminator models
    netD = LSTMDiscriminator(in_dim=in_dim, device=device).to(device)
    netG = LSTMGenerator(in_dim=in_dim, out_dim=in_dim, device=device).to(device)

    #print("|Discriminator Architecture|\n", netD)
    #print("|Generator Architecture|\n", netG)
    #print("")


    # Setup loss function
    criterion = nn.BCELoss().to(device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt_trn.lr)
    optimizerG = optim.Adam(netG.parameters(), lr=opt_trn.lr)

    #Adversarial Training of Generator and Discriminator models
    real_label = 1
    fake_label = 0

    D_loss_list = []
    G_loss_list = []
    # for epoch in range(opt_trn.epochs):

    

    for epoch in range(opt_trn.epochs-7):
        for i, (x, y) in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with real data
            netD.zero_grad()
            real = x.to(device)  # shape:(32,48)
            # print(real)
            # real.unsqueeze_(-1)
            batch_size, seq_len = real.size(0), real.size(1)
            # real = real.expand(batch_size, seq_len, 1)
            # real = torch.full((batch_size, seq_len, 1), real, device = device)

            label = torch.full((batch_size, seq_len, 1), real_label, device=device)

            # 2차원의 real을 3차원으로 만들기
            real.unsqueeze_(-1)
            batch_size, seq_len = real.size(0), real.size(1)
            real = real.expand(batch_size, seq_len, 1)

            output, _ = netD.forward(real)  # 3-D여야 하는데 got 2?

            errD_real = criterion(output.float(), label.float())
            errD_real.backward()
            optimizerD.step()
            D_x = output.mean().item()

            # Train with fake data??? 더 D와 G를 똑똑하게 하려고??
            noise = Variable(init.normal(torch.Tensor(batch_size, seq_len, in_dim), mean=0, std=0.1)).cuda()
            # noise = Variable(init.normal(torch.Tensor(batch_size,seq_len,in_dim),mean=0,std=0.1))
            fake, _ = netG.forward(noise)
            output, _ = netD.forward(
                fake.detach())  # detach causes gradient is no longer being computed or stored to save memeory
            label.fill_(fake_label)
            errD_fake = criterion(output.float(), label.float())
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            noise = Variable(init.normal(torch.Tensor(batch_size, seq_len, in_dim), mean=0, std=0.1)).cuda()
            # noise = Variable(init.normal(torch.Tensor(batch_size,seq_len,in_dim),mean=0,std=0.1))
            fake, _ = netG.forward(noise)
            label.fill_(real_label)
            output, _ = netD.forward(fake)
            errG = criterion(output.float(), label.float())
            errG.backward()
            optimizerG.step()
            D_G_z2 = output.mean().item()
            D_loss_list.append(errD.item())
            G_loss_list.append(errG.item())

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
               % (epoch, opt_trn.epochs, i, len(dataloader),
                  errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='')
        print()

    ###################################################################################################
    ###Anomaly Detection###
    print('start anomaly detection')

    #Define basic settings for inverse mapping
    class ArgsTest:
        workers = 1
        batch_size = 1


    opt_test = ArgsTest()

    generator = netG # changing reference variable
    discriminator = netD # changing reference variable

    # location of datasets and category
    #end_name = 'cpu_utilization_asg_misconfiguration.csv' # dataset name
    end_name = '하얀노트북/split with seed/changui_test.csv' #ISSDA의 80%인 training data
    data_file = '하얀노트북/'+end_name # dataset category and dataset name
    key = 'realKnownCause/'+end_name # This key is used for reading anomaly labels

    #define test data
    # Define settings for loading data in evaluation mood
    class TestDataSettings:

        def __init__(self):
            # self.BASE = 'D:\\ResearchDataGtx1060\\AnomalyDetectionData\\NabDataset\\'
            self.BASE = '/content/drive/My Drive/'
            # self.label_file = 'labels\\combined_windows.json'
            self.data_file = data_file
            self.key = key
            self.train = False  # 이거만 training이랑 다른점


    test_data_settings = TestDataSettings()

    # define dataset object and data loader object in evaluation mood for NAB dataset

    test_dataset = NabDataset(test_data_settings)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt_test.batch_size,
                                             shuffle=False, num_workers=int(opt_test.workers))

    test_dataset.x.shape, test_dataset.y.shape, test_dataset.data_len # check the dataset shape

    #define a function to calculate anomaly score
    # Lambda = 0.1 according to paper
    # x is new data, G_z is closely regenerated data

    def Anomaly_score(x, G_z, Lambda=0.1):
        residual_loss = torch.sum(torch.abs(x - G_z))  # Residual Loss

        # x_feature is a rich intermediate feature representation for real data x
        real = x.to(device)

        if len(list(real.size())) == 2:
            # print(real.shape)
            real.unsqueeze_(-1)
            batch_size, seq_len = real.size(0), real.size(1)
            real = real.expand(batch_size, seq_len, 1)
            # print(real.shape)
            output, x_feature = discriminator(x.to(device))
            # output, x_feature = discriminator(real)

        else:
            output, x_feature = discriminator(real)
        # G_z_feature is a rich intermediate feature representation for fake data G(z)
        output, G_z_feature = discriminator(G_z.to(device))

        discrimination_loss = torch.sum(torch.abs(x_feature - G_z_feature))  # Discrimination loss

        total_loss = (1 - Lambda) * residual_loss.to(device) + Lambda * discrimination_loss
        return total_loss

    #inverse mapping to latent space and reconstruction of data for estimating anomaly score
    loss_list = []
    # y_list = []
    for i, (x, y) in enumerate(test_dataloader):
        # print(i, y)

        z = Variable(init.normal(torch.zeros(opt_test.batch_size,
                                             test_dataset.window_length,
                                             test_dataset.n_feature), mean=0, std=0.1), requires_grad=True)
        # z = x
        z_optimizer = torch.optim.Adam([z], lr=1e-2)

        loss = None
        for j in range(50):  # set your interation range
            gen_fake, _ = generator(z.cuda())
            # gen_fake,_ = generator(z)
            loss = Anomaly_score(Variable(x).cuda(), gen_fake)
            # loss = Anomaly_score(Variable(x), gen_fake)
            loss.backward()
            z_optimizer.step()

        loss_list.append(loss)  # Store the loss from the final iteration
        # y_list.append(y) # Store the corresponding anomaly label
        print('~~~~~~~~loss={},  y={} ~~~~~~~~~~'.format(loss, y))
        # break

    #################################################################################################################3
    ###성능 내기###
    def roc(loss_list, threshold):
        test_score_df = pd.DataFrame(index=range(len(loss_list)))
        test_score_df['loss'] = [loss.item() / test_dataset.window_length for loss in loss_list]  # 29027
        test_score_df['y'] = test_dataset.y
        test_score_df['threshold'] = threshold
        test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
        test_score_df['t'] = [x[47].item() for x in test_dataset.x]  # x[59]

        start_end = []
        state = 0
        for idx in test_score_df.index:
            if state == 0 and test_score_df.loc[idx, 'y'] == 1:
                state = 1
                start = idx
            if state == 1 and test_score_df.loc[idx, 'y'] == 0:
                state = 0
                end = idx
                start_end.append((start, end))

        for s_e in start_end:
            if sum(test_score_df[s_e[0]:s_e[1] + 1]['anomaly']) > 0:
                for i in range(s_e[0], s_e[1] + 1):
                    test_score_df.loc[i, 'anomaly'] = 1

        actual = np.array(test_score_df['y'])
        predicted = np.array([int(a) for a in test_score_df['anomaly']])

        return actual, predicted


    # AUROC 구하기
    import pandas as pd
    import numpy as np

    # threshold = 1,2,3....100으로 해주기
    threshold_list = []
    for i in range(100):
        threshold_list.append(i)

    final_actual11 = []
    final_predicted11 = []

    TPR = []
    FPR = []

    for i in range(len(threshold_list)):
        ac, pr = roc(loss_list, threshold_list[i])
        final_actual11.append(ac)
        final_predicted11.append(pr)

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        # compare final_actual11[i] and final_predicted11[i]
        for j in range(len(final_actual11[i])):
            if final_actual11[i][j] == 1 and final_predicted11[i][j] == 1:
                TP += 1
            elif final_actual11[i][j] == 1 and final_predicted11[i][j] == 0:
                FN += 1
            elif final_actual11[i][j] == 0 and final_predicted11[i][j] == 1:
                FP += 1
            elif final_actual11[i][j] == 0 and final_predicted11[i][j] == 0:
                TN += 1

        TPR.append(TP / (TP + FN))
        FPR.append(FP / (FP + TN))

    """    
    #TPR, FPR에 (0,0),(1,1)넣기
    TPR.insert(0,0)
    TPR.append(1)
    FPR.insert(0,0)
    FPR.append(1)
    """

    # 최종 면적 구하기
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc

    plt.plot(FPR, TPR)
    area = auc(FPR, TPR)

    print('area under curve:', area)

    #############################################################################################################
    ###저장공간 비우기###
    del(loss_list)
    del(D_loss_list)
    del(G_loss_list)


    area_list.append(area)
print(area_list)

