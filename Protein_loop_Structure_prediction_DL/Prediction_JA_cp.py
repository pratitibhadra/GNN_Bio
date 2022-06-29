import time
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import SGD


def data_read(folder,target_fl):

    dl_dir = folder #r'./csv/'  # all target pdb should keep in the folder

    csvs = [x for x in os.listdir(dl_dir) if os.path.splitext(x)[-1] == '.csv']

    data_target_df = pd.read_csv(target_fl,sep=",")
    
    y=[] 
    csv_cnt = 0
    for fl in csvs:

        fname = folder + fl
        data2D = np.around(np.loadtxt(open(fname, "rb"), delimiter=",",skiprows=1),decimals = 3)
        
        if csv_cnt == 0:
            X = np.reshape(data2D,(1,data2D.shape[0],data2D.shape[1]))
        else:
            data3D = np.reshape(data2D,(1,data2D.shape[0],data2D.shape[1]))
            tempX = np.append(X,data3D,axis=0)
            X = tempX
  
        csv_cnt = csv_cnt+1

        S_ID = fl.split('_')[0] + '_' + fl.split('_')[1]
        for i,val in enumerate(data_target_df["Serial_ID"]):
            if val == S_ID:
                y.append(data_target_df["sJA"][i]*np.pi/180)
                #print(val)

    #df_y = pd.DataFrame(y, columns = ['sJA'])
    #sin_sJA = np.sin( df_y['sJA'].values * np.pi/180)
    #cos_sJA = np.cos(df_y['sJA'].values * np.pi/180)
    
    #targ=np.vstack((sin_sJA,cos_sJA)) 
    
    #target_scaler = MinMaxScaler(feature_range=(-1, 1))
    #targ = target_scaler.fit_transform(y)

    #print(targ)
    #exit()

    
    #targ[1]=cos_sJA
    #print(targ[0,:])
    #exit()
    #temp_X = X[:,:,1:];
    #temp_y = 
    #X.reshape(-1,X.shape[1],X.shape[2],1)

    return{
            'feature': X[:,:,1:],
            'target': np.array(y)
        }


def define_model(data):
    model = Sequential()

    ## -- NN1
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(data.shape[1], data.shape[2], 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='softmax'))
    # compile model
    #opt = SGD(lr=0.01, momentum=0.9)
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(loss='mean_absolute_error', optimizer=opt)
    
    ## -- NN2
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(MaxPooling2D((2, 2),padding='same'))
    #model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    #model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    #model.add(Flatten())
    #model.add(Dense(128, activation='linear'))
    #model.add(LeakyReLU(alpha=0.1))
    #model.add(Dense(1, activation='softmax'))
    
    
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

#def evaluate_model(dataX,datay,n_folds):
#    print(dataX)
#    print(datay)
#    scores, histories = list(), list()
    # prepare cross validation
#    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
#    for train_ix, test_ix in kfold.split(dataX):
#        model=define_model(dataX)
        
    
#def test_model():

if __name__ == '__main__':
    dataset = data_read('./HHC_newpssm/','HHC_target_vari.csv')
    X = dataset['feature']
    X = X.reshape(-1,X.shape[1],X.shape[2],1)
    y = dataset['target']
    y = y.reshape(y.shape[0],1)

    rn=random.randint(1,50)
    train_X,test_X,train_y,test_y = train_test_split(X, y, test_size=0.2, random_state=rn)

    
    # prepare cross validation
    n_folds = 5
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(train_X):
        model=define_model(train_X)
        # select rows for train and test
	    #tnX=train_X[train_ix]
            #tnY=train_y[train_ix]
            #testX = trian_X[test_ix]
            #testY = train_y[test_ix]
	# fit model
            #print(train_ix)
            #print(test_ix)
            #print(train_X[train_ix])
        model.fit(train_X[train_ix], train_y[train_ix], epochs=10, batch_size=32, validation_data=(train_X[test_ix], train_y[test_ix]), verbose=0)
        score=model.evaluate(train_X[test_ix], train_y[test_ix], verbose=0)
        print(score)
        #print(score[1])
    #scores,histories = evaluate_model(train_X,train_y,5)
    #train_X,test_X,train_y,test_y = train_test_split(X, y, test_size=0.2, random_state=13)


#print(X)

#print(X[1])
