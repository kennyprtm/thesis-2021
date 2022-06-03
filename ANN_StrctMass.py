#%%

#Artificial Neural Network

PYTHONHASHSEED = 0
#Library Importing 
import pandas as pd
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import set_random_seed
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

#To teproduce the result
seed(132)
set_random_seed(45)

os.chdir('E:\ITB\AEROSPACE_ENGINEERING\#S2\Code')

dataset = pd.read_excel('ANN_Data.xlsx', sheet_name='Struct_Mass')
x = dataset[['Plies', 'Relative_Density', 'Material', 'AB_Ratio', 'HC_Ratio',
             'Geometry']]
Y = dataset[['Mass']].values.astype(np.float64)

#Data Preprocessing
one_hot_mat = pd.get_dummies(x['Material'])
x = x.drop('Material', axis=1)
x = x.join(one_hot_mat)

one_hot_geom = pd.get_dummies(x['Geometry'])
x = x.drop('Geometry', axis=1)
x = x.join(one_hot_geom)

max_abs_scaler = MaxAbsScaler()
Y = max_abs_scaler.fit_transform(Y)

#%%
#Split Data to Training Data and Testing Data
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.2, 
                                                    random_state=0, 
                                                    shuffle=True)

#%%
#Scatter 3D
#fig = py.figure().gca(projection='3d')
#fig.scatter(x['Relative_Density'], x['Plies'], Y)
#fig.plot()

#%%
#Artificial Neural Network Model
model = Sequential()
model.add(Dense(units = 12, input_dim = 12, kernel_initializer='uniform', 
                activation='relu', use_bias=True))
model.add(Dense(units = 8, kernel_initializer='uniform', 
                activation='relu', use_bias=True))
model.add(Dense(units = 8, kernel_initializer='uniform', 
                activation='relu', use_bias=True))
model.add(Dense(units = 1, kernel_initializer='uniform', 
                activation='linear', use_bias=True))
model.compile(optimizer='adam', loss='mse', metrics=['mape'])

#Model Summary
model.summary()

#Model Training
history = model.fit(x_train,Y_train,validation_data=(x_test, Y_test), 
                    epochs=1000,batch_size=4)

#Model Evaluation
scores = model.evaluate(x_test,Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))

#Plot Accuracy
py.plot(history.history['loss'], label='train')
py.plot(history.history['val_loss'], label='test')
py.xlabel('Iteration')
py.ylim([-0.02, 0.3])
py.ylabel('Loss (MSE)')
py.legend()
py.show()

#%%
#Save Model
model.save('model_mass.h5')
print('Model Saved to Disk')

#Save Data Test for Model Evaluation in NSGA-II Optimization
path = r'Data_Test_Mass.xlsx'
writer = pd.ExcelWriter(path, engine='openpyxl')

dfx = pd.DataFrame(x_test)
dfx.to_excel(writer, sheet_name='x_test')

dfy = pd.DataFrame(Y_test)
dfy.to_excel(writer, sheet_name='Y_test')

writer.save()
writer.close()