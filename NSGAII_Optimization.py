#%%
#NSGA-II Optimization

from tensorflow.keras.models import load_model
from platypus import NSGAII, Problem, Real, Integer
from platypus.operators import CompoundOperator, SBX, PM, HUX, BitFlip
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('E:\ITB\#S2\Code')

#load_model
model_mass = load_model('model_mass.h5')
model_deform = load_model('model_deform.h5')

#Model Summary Evaluation
model_mass.summary()
model_deform.summary()

# #%%
# #Model for Evaluation Import
dfx_mass = pd.read_excel('Data_Test_Mass.xlsx', sheet_name='x_test')
dfy_mass = pd.read_excel('Data_Test_Mass.xlsx', sheet_name='Y_test')
dfx_deform = pd.read_excel('Data_Test_Deform.xlsx', sheet_name='x_test')
dfy_deform = pd.read_excel('Data_Test_Deform.xlsx', sheet_name='Y_test')

x_test_mass = dfx_mass.iloc[:,1:13].values
y_test_mass = dfy_mass.iloc[:,1].values
x_test_deform = dfx_deform.iloc[:,1:13].values
y_test_deform = dfy_deform.iloc[:,1].values

# #Model Evaluation
model_mass.evaluate(x_test_mass,y_test_mass)
model_deform.evaluate(x_test_deform,y_test_deform)

#%%
#Scaling Data for Inverse Scaling
os.chdir('E:\ITB\#S2\Code')

#For Structural Mass
dataset = pd.read_excel('ANN_Data.xlsx', sheet_name='Struct_Mass')
Y_mass = dataset[['Mass']].values.astype(np.float64)
max_abs_scaler_mass = MaxAbsScaler()
scaling_mass = max_abs_scaler_mass.fit_transform(Y_mass)

dataset = pd.read_excel('ANN_Data.xlsx', sheet_name='Batt_Deform')
Y_deform = dataset[['Batt_Deform']].values.astype(np.float64)
max_abs_scaler_deform = MaxAbsScaler()
scaling_deform = max_abs_scaler_deform.fit_transform(Y_deform)

#%%

#Calculate Scaled Value of Maximum Deformation Allowed
maxdeformscaled = max_abs_scaler_deform.transform(np.array([[3]]))

#%%
def massdeform (vars):
#Variable Definition 
    Plies = vars[0]
    Relative_Density = vars[1]
    AB_Ratio = vars[2]
    HC_Ratio = vars[3]
    AlSi12 = vars[4]
    Nylon = vars[5]
    PLA = vars[6]
    Ti6Al4V = vars[7]
    BCC = vars[8]
    BCCZ = vars[9]
    OC = vars[10]
    OT = vars[11]

#Input Variable Matrix Definition
    IntVar = np.array([[Plies, Relative_Density, AB_Ratio, HC_Ratio, AlSi12, 
                        Nylon, PLA, Ti6Al4V, BCC, BCCZ, OC, OT]])
    
#Function Definition 
    mass = model_mass.predict(IntVar)
    deform = model_deform.predict(IntVar)
    
#Return
    return [mass.item(), deform.item()], [AlSi12+Nylon+PLA+Ti6Al4V-1, 
                                          BCC+BCCZ+OC+OT-1, 
                                          deform-maxdeformscaled]


#%%
#Define Problem

igr13 = Integer(1,3)
igr01 = Integer(0,1)

problem = Problem(12, 2, 3)
problem.types[:] = [igr13, Real(0.1,0.25), Real(0.7,1), Real(0.75,1.5), igr01, 
                    igr01, igr01, igr01, igr01, igr01, igr01, igr01]
problem.constraints[0:1] = "==0"
problem.constraints[2] = "<0"
problem.function = massdeform

algorithm = NSGAII(problem, variator = CompoundOperator(SBX(), HUX(), 
                                                        BitFlip(), PM()))
algorithm.run(20000)

result = algorithm.result
feasible_solutions = [s for s in result if s.feasible]
for solution in feasible_solutions:
    print(solution.objectives)

x = [s.objectives[0] for s in algorithm.result]
y = [s.objectives[1] for s in algorithm.result]
plt.scatter(x,y)
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
plt.xlabel("Cell Mass x Plies (Scaled)")
plt.ylabel("Battery Deformation (Scaled)")
plt.show()


#%%
#Decision Making - TOPSIS

n = len(x)

C1 = []
C2 = []
for i in range(0,n):
    C1.append(float(x[i]))
    C2.append(float(y[i]))

C1 = np.array(C1)
C2 = np.array(C2)

Categories = np.transpose(np.array((C1,C2)))

#Sum Denominator
sum_denom_mass = 0
sum_denom_deform = 0
for i in range (0,n):
    sum_denom_mass = sum_denom_mass + pow(Categories[i][0],2)
    sum_denom_deform = sum_denom_deform + pow(Categories[i][1],2)
    
denom_mass = pow(sum_denom_mass,0.5)
denom_deform = pow(sum_denom_deform,0.5)

#Calculate Normalized Value
Norm_Matrix = np.zeros(np.shape(Categories))
for i in range (0,n):
    Norm_Matrix[i][0] = Categories[i][0]/denom_mass
    Norm_Matrix[i][1] = Categories[i][1]/denom_deform
    
#Weighted Normalized Matrix 
mass_weight = 0.5
deform_weight = 0.5

Norm_Weighted = np.zeros(np.shape(Categories))
for i in range(0,n):
    Norm_Weighted[i][0] = Norm_Matrix[i][0]*mass_weight
    Norm_Weighted[i][1] = Norm_Matrix[i][1]*deform_weight

#Finding Maximum and Minimum Value
max_mass = Norm_Weighted[i][0]
max_deform = Norm_Weighted[i][1]
min_mass = Norm_Weighted[i][0]
min_deform = Norm_Weighted[i][1]
for i in range(0,n):
    if Norm_Weighted[i][0] > max_mass:
        max_mass = Norm_Weighted[i][0]
    if Norm_Weighted[i][1] > max_deform:
        max_deform = Norm_Weighted[i][1]
    if Norm_Weighted[i][0] < min_mass:
        min_mass = Norm_Weighted[i][0]
    if Norm_Weighted[i][1] < min_deform:
        min_deform = Norm_Weighted[i][1]      

#Positive and Negatve Value
A_Pos = [min_mass, min_deform]
A_Neg = [max_mass, max_deform]

#Separation or Distance Matrix
#Column 1 for Ideal, Column 2 for Negative Ideal
ny = np.shape(Categories)[1]

S = np.zeros((n,2))
for i in range(0,n):
    for j in range(0,ny):
        S[i][0] = S[i][0] + pow((A_Pos[j] - Norm_Weighted[i][j]),2)
        S[i][1] = S[i][1] + pow((A_Neg[j] - Norm_Weighted[i][j]),2)
    S[i] = pow(S[i],0.5)
    
#Closeness
Cl = np.zeros(np.shape(Categories)[0])
for i in range(0,n):
    Cl[i] = S[i][1]/(S[i][0]+S[i][1])
    
#Find Maximum Closeness
Max_Cl = Cl[0] #Initial Value
idx = 0
for i in range(0,n):
    if Cl[i]>Max_Cl:
        Max_Cl = Cl[i]
        idx = i

#Best Result Extraction
Best_Opt = [x[idx], y[idx]]
Best_Opt = [max_abs_scaler_mass.inverse_transform(np.array([[Best_Opt[0]]])),
            max_abs_scaler_deform.inverse_transform(np.array([[Best_Opt[1]]]))]

#Display Prediction
pred_df = pd.DataFrame(np.reshape(Best_Opt, (1, len(Best_Opt))), 
                       columns=['Mass', 'Batt_Deform'], index=['Prediction'])
print(pred_df.T)

#Obtain Input Parameter
Input_Param = [s.variables for s in feasible_solutions]
Input_Param = Input_Param[idx]

# #Decode Input Parameter
# for i in range(0,len(Input_Param)):
#     temp_Input_Param = algorithm.result[i].variables[i]
#     Input_Param[i] = problem.types[i].decode(temp_Input_Param)

#Display Input Parameters
column_name = list(dfx_mass.columns)
del column_name[0]

result_df = pd.DataFrame(np.reshape(Input_Param, (1,len(Input_Param))), 
                         columns=column_name, index=['Input Parameter'])
print(result_df.T)

#%%
#Save Result
# path = r'NSGA_Result.xlsx'
# writer = pd.ExcelWriter(path, engine='openpyxl')
    
# nsgaresult = pd.DataFrame(list(zip(x, y)),
#                 columns =['Cell Mass x Plies (Scaled)', 'Battery Deformation'])
# nsgaresult.to_excel(writer, sheet_name='Result')

# writer.save()
# writer.close()

# print('NSGA Data Saved to Disk')