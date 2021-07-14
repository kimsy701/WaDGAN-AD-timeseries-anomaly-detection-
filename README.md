# timeseries-anomaly-detection

### WaDGAN-AD   
### (advanced model of LSTM-GAN by structural changes in hidden layer LSTM)

### 1. Dataset   
#### 1.1 Real dataset   
  -splited to train, validation, test set   
  -train dataset has no anomalies, validation and test dataset has 1% of anomalies   
#### 1.2 Synthetic dataset   
  -made to demonstrate structural changes in WaDGAN-AD   
  -each dataset has one cycle(cycle of year), two cycles(cycle of year, week), three cycles(cycle of year, week, day) respectively   
  -three synthetic dataset   
  -spliteed to train, validation, test set   
  -train dataset has no anomalies, validation and test dataset has 1% of anomalies   
  -made by make synthetic(3).py   
  
### 2. Data Preprocessing   
#### 2.1 preprocess/koreaunivdata_preprocessing.py   
  -split data to train, validation, test set and give abnoraml data to validation and test set   
  
### 3. Model   
#### 3.1 main/nab_dataset.py   
  -bring dataset and make it to 3d to fit into deep-learning models such as LSTM   
#### 3.2 models/recurrent_models_pyramid_{2}.py   
  -LSTM generator and LSTM discriminator of WaDGAN-AD.    
#### 3.3 main/data{A}_model{A}.py   
  -execute WaDGAN-AD   

