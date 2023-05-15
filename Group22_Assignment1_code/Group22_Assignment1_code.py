#%%
# importing modules 
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#%%


# reading data for question 1 i.e. linearly saperable data using perceptron 
data_q1_c1 = pd.read_csv("Classification/LS_Group22/Class1.txt",index_col= False,names=["X1", "X2"], delimiter=" ")
data_q1_c1["Y"] = 1   # appending class for future reference 
data_q1_c2 = pd.read_csv("Classification/LS_Group22/Class2.txt",index_col = False,names=["X1", "X2"], delimiter=" ")
data_q1_c2["Y"] = 2
data_q1_c3 = pd.read_csv("Classification/LS_Group22/Class3.txt",index_col = False,names=["X1", "X2"], delimiter=" ")
data_q1_c3["Y"] = 3
all_data = pd.concat([data_q1_c1,data_q1_c2,data_q1_c3])

# train test split 
data_q1_c1_x_train,data_q1_c1_x_test,data_q1_c1_y_train,data_q1_c1_y_test = train_test_split(data_q1_c1.iloc[:,:-1],data_q1_c1.iloc[:,2:],test_size = 0.3,random_state =0)
data_q1_c2_x_train,data_q1_c2_x_test,data_q1_c2_y_train,data_q1_c2_y_test = train_test_split(data_q1_c2.iloc[:,:-1],data_q1_c2.iloc[:,2:], test_size = 0.3,random_state =0)
data_q1_c3_x_train,data_q1_c3_x_test,data_q1_c3_y_train,data_q1_c3_y_test = train_test_split(data_q1_c3.iloc[:,:-1],data_q1_c3.iloc[:,2:],test_size = 0.3 ,random_state =0)


#%%

def perceptron (x_train , y_train ):
    # x_train = pd.DataFrame(x_train)
    # y_train = pd.DataFrame(y_train)
    threshold = 0.001
    w = []
    d = len(x_train.iloc[0])
    # max_x_train= max(x_train.values)
    # min_x_train = min(x_train.values)
    epoch_error = []
    last_error = -1

    # step 1: initialise w
    for i in range(d):
        w.append(0)
    w.insert(0,0)  # annote w0
    converged = False
    neta = 0.009

    while (not converged):
        n = len(x_train)
        avg_error = 0
        for i in range(n):
            # selecting training example
            xn = x_train.iloc[i]
            yn = y_train.iloc[i]
            activation_a = 0 
            bias = 0

            # step 3 computing the output of the neuron
            for j in range(d+1):
                if (j==0):
                    activation_a = activation_a + w[j]
                else:
                    activation_a  = activation_a + xn[j-1]*w[j]
            
            # step 4 computing instantaneous error En
            sn = (1/(1+np.exp(-1*activation_a)))
            En = 0.5*(yn-sn)**2

            #step 5 calculating delta w to update the weights 
            ded = float((neta)*(yn-sn)*(sn)*(1-sn))

            xn = xn*ded
            xn = list(xn)
            xn.insert(0,ded)
          
            # step 6 updating the w (parameters)
            for k in range(len(w)):
                w[k] = w[k] + xn[k]
            bias = bias + ded
            avg_error += En
        # calculating the avg error
        avg_error = float(avg_error/float(n))
        epoch_error.append(avg_error)

        # calculating the delta error to check convergence 
        delta_error = float(last_error - avg_error)
        if (last_error !=1 and (delta_error<= threshold) and (delta_error>0)):
            converged = True
        last_error = avg_error
    
    #plotting the graphs
    plt.title("Average Error vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Average Squared Error (En)")
    plt.plot(np.arange(1,len(epoch_error)+1), epoch_error)
    plt.show()
    return w

#training models 
# concating the data set of two class for 1 vs 1 perceptron model and doing same for all the permutation and combinations 
x_train_c1_c2 = pd.concat([data_q1_c1_x_train,data_q1_c2_x_train])
y_train_c1_c2 =[]
for i in range(len(data_q1_c1_x_train)+len(data_q1_c2_x_train)):

    # appending the yn vector and repeating the same in below cases 
    if ( i < len(data_q1_c1_x_train)):
        y_train_c1_c2.append(1)
    else:
        y_train_c1_c2.append(0)
y_train_c1_c2 = pd.DataFrame(y_train_c1_c2)

# w parameters from the function of perceptron 
w_c1_c2 = perceptron(x_train_c1_c2,y_train_c1_c2)


x_train_c1_c3 = pd.concat([data_q1_c1_x_train,data_q1_c3_x_train])
y_train_c1_c3 =[]
for i in range(len(data_q1_c1_x_train)+len(data_q1_c3_x_train)):
    if ( i < len(data_q1_c1_x_train)):
        y_train_c1_c3.append(1)
    else:
        y_train_c1_c3.append(0)
y_train_c1_c3 = pd.DataFrame(y_train_c1_c3)

# w parameters from the function of perceptron 
w_c1_c3 = perceptron(x_train_c1_c3,y_train_c1_c3)


x_train_c2_c3 = pd.concat([data_q1_c2_x_train,data_q1_c3_x_train])
y_train_c2_c3 =[]
for i in range(len(data_q1_c2_x_train)+len(data_q1_c3_x_train)):
    if ( i < len(data_q1_c2_x_train)):
        y_train_c2_c3.append(1)
    else:
        y_train_c2_c3.append(0)
y_train_c2_c3 = pd.DataFrame(y_train_c2_c3)

# w parameters from the function of perceptron 
w_c2_c3 = perceptron(x_train_c2_c3,y_train_c2_c3)


        
# %%

# test of question 1
def predict_class(test_data):
    y_preds = []
    for i in range(len(test_data)):
        x_test = list(test_data.iloc[i])
        x_test.insert(0,1)
        
        output = 0
        c_1v_2 =0
        c_1v_3 =0
        c_2v_3 =0
        
        #computing the output of the neuron wrt all the models trained

        for j in range(len(x_test)):
            output  = output + w_c1_c2[j]*x_test[j]
        s = 1/(1+np.exp((-1)*output))
        if (s>0.5):
            c_1v_2 = 1

        output=0
        for j in range(len(x_test)):
            output  = output + w_c1_c3[j]*x_test[j]
        s = 1/(1+np.exp((-1)*output))
        
        if (s>0.5):
            c_1v_3 = 1

        output = 0
        for j in range(len(x_test)):
            output  = output + w_c2_c3[j]*x_test[j]
        s = 1/(1+np.exp((-1)*output))
        if (s>0.5):
            c_2v_3 = 1
        
        # appending the class on basis of majority (majority wins)
        if(c_1v_2 == 1  and c_1v_3 == 1):
            y_preds.append(1)
        elif(c_1v_2 == 0  and c_2v_3 == 1):
            y_preds.append(2)
        elif(c_1v_3 ==0 and c_2v_3 == 0):
            y_preds.append(3)  
    return y_preds

# %%

# taking predictions on the test data 
y_pred_c1 = list(predict_class(data_q1_c1_x_test))
y_pred_c2 =list(predict_class(data_q1_c2_x_test))
y_pred_c3 =list(predict_class(data_q1_c3_x_test))

# concating the predictions into one data frame to claculate the accuracy 
y_pred = y_pred_c1+y_pred_c2+y_pred_c3
# concating the known class into one data frame to claculate the accuracy 
y_test = pd.concat([data_q1_c1_y_test,data_q1_c2_y_test,data_q1_c3_y_test])


# %%
# importing modules for calculations 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# printing the confusion matrix and accuracy score 
print("Confusion matrix: \n " ,confusion_matrix(y_test, y_pred))
print("Accuracy of the model is " ,accuracy_score(y_test, y_pred))


# %%
# plotting the descision boundary plot of the linearly saperable data 

fig, axes = plt.subplots()
axes.scatter(data_q1_c1_x_train['X1'], data_q1_c1_y_train['Y'], label = 'class 1')
axes.scatter(data_q1_c2_x_train['X1'], data_q1_c2_y_train['Y'], label = 'class 2')
#print(data_q1_c2)
all_data = pd.concat([data_q1_c1,data_q1_c2,data_q1_c3])

x_min, x_max = all_data['X1'].min() -1, all_data['X1'].max() + 1
y_min, y_max = all_data['X2'].min() - 1, all_data['X2'].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
z = np.array([xx.ravel(), yy.ravel()]).T
Z=np.array(predict_class(pd.DataFrame(z)))
Zx=[0 for x in range(76608-len(Z))]
Z=np.append(Z,Zx)
Z=Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha = 0.3)


# Plot the decision regions

plt.scatter(data_q1_c1['X1'],data_q1_c1['X2'])
plt.scatter(data_q1_c2['X1'],data_q1_c2['X2'])
plt.scatter(data_q1_c3['X1'],data_q1_c3['X2'])
plt.title('decision region')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.show()

# %%
# initialising the lists to append the data of different classes 
data_q2_c1 = []
data_q2_c2 = []
data_q2_3 = []
data_q2 = []
# reading the data 
with open("Classification/NLS_Group22.txt",'r') as f:
    a = f.read().split()
l = []
for i in range(0, len(a)):
    l.append(float(a[i]))
    if i % 2 == 1:
        # l.append(float(a[i]))
        data_q2.append(l)
        l = []
        continue
data_q2_c1=pd.DataFrame(data_q2[0:500])
data_q2_c1["y"] = 1 # appending the class for future reference
data_q2_c2=pd.DataFrame(data_q2[500:1000])
data_q2_c2["y"] = 2
data_q2_c3=pd.DataFrame(data_q2[1000:])
data_q2_c3["y"] = 3

    


#%%
# train test split
data_q2_c1_x_train,data_q2_c1_x_test,data_q2_c1_y_train,data_q2_c1_y_test = train_test_split(data_q2_c1.iloc[:,:-1],data_q2_c1.iloc[:,2:],test_size = 0.3,random_state =42)
data_q2_c2_x_train,data_q2_c2_x_test,data_q2_c2_y_train,data_q2_c2_y_test = train_test_split(data_q2_c2.iloc[:,:-1],data_q2_c2.iloc[:,2:], test_size = 0.3,random_state =42)
data_q2_c3_x_train,data_q2_c3_x_test,data_q2_c3_y_train,data_q2_c3_y_test = train_test_split(data_q2_c3.iloc[:,:-1],data_q2_c3.iloc[:,2:],test_size = 0.3,random_state =42 )

#training models 
# concating the data set of two class for 1 vs 1 perceptron model and doing same for all the permutation and combinations 

x_train_c1_c2 = pd.concat([data_q2_c1_x_train,data_q2_c2_x_train])
y_train_c1_c2 =[]
for i in range(len(data_q2_c1_x_train)+len(data_q2_c2_x_train)):
    # appending the yn vector and repeating the same in below cases 
    if ( i < len(data_q2_c1_x_train)):
        y_train_c1_c2.append(1)
    else:
        y_train_c1_c2.append(0)
y_train_c1_c2 = pd.DataFrame(y_train_c1_c2)

# w parameters from the function of perceptron 
w_c1_c2 = perceptron(x_train_c1_c2,y_train_c1_c2)


x_train_c1_c3 = pd.concat([data_q2_c1_x_train,data_q2_c3_x_train])
y_train_c1_c3 =[]
for i in range(len(data_q2_c1_x_train)+len(data_q2_c3_x_train)):
    if ( i < len(data_q2_c1_x_train)):
        y_train_c1_c3.append(1)
    else:
        y_train_c1_c3.append(0)
y_train_c1_c3 = pd.DataFrame(y_train_c1_c3)

# w parameters from the function of perceptron 
w_c1_c3 = perceptron(x_train_c1_c3,y_train_c1_c3)


x_train_c2_c3 = pd.concat([data_q2_c2_x_train,data_q2_c3_x_train])
y_train_c2_c3 =[]
for i in range(len(data_q2_c2_x_train)+len(data_q2_c3_x_train)):
    if ( i < len(data_q2_c2_x_train)):
        y_train_c2_c3.append(1)
    else:
        y_train_c2_c3.append(0)
y_train_c2_c3 = pd.DataFrame(y_train_c2_c3)

# w parameters from the function of perceptron 
w_c2_c3 = perceptron(x_train_c2_c3,y_train_c2_c3)

#%%
# taking predictions on the test data 
y_pred_c1 = list(predict_class(data_q2_c1_x_test))
y_pred_c2 =list(predict_class(data_q2_c2_x_test))
y_pred_c3 =list(predict_class(data_q2_c3_x_test))

#concating the predictions for accuracy calculations 
y_pred = y_pred_c1+y_pred_c2+y_pred_c3
#concating the known classes for accuracy calculations 
y_test = pd.concat([data_q2_c1_y_test,data_q2_c2_y_test,data_q2_c3_y_test])


# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#printing the accuracy score and confusion matrices
print("Confusion matrix: \n " ,confusion_matrix(y_test, y_pred))
print("Accuracy of the model is " ,accuracy_score(y_test, y_pred))


############################################################################################
################################   Regression   ############################################
############################################################################################

# %%
# reading the data 
dataset = pd.read_csv("Regression/UnivariateData/22.csv", names = ["X", "Y"])
# train test split 
X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:,:-1],dataset.iloc[:,-1] , test_size=0.3, shuffle=False)
plt.scatter(X_train,y_train)
plt.show()
#%%

def perceptron (x_train , y_train ):
    # x_train = pd.DataFrame(x_train)
    # y_train = pd.DataFrame(y_train)
    threshold = 0.001
    w = []
    d = len(x_train.iloc[0])
    # max_x_train= max(x_train.values)
    # min_x_train = min(x_train.values)
    epoch_error = []
    last_error = -1

    # step 1 initialising the w vector 
    for i in range(d):
        w.append(0)
    # annoting the w0
    w.insert(0,0)
    converged = False
    neta = 0.009
    
    while (not converged):
        n = len(x_train)
        avg_error = 0
        
        for i in range(n):
            # step 2  choosing training example 
            xn = x_train.iloc[i]
            yn = y_train.iloc[i]
            activation_a = 0
            bias = 0
            # step 3 computing the ouptut of neuron
            for j in range(d+1):
                if (j==0):
                    activation_a = activation_a + w[j]
                else:
                    activation_a  = activation_a + xn[j-1]*w[j]
            sn = activation_a
            # calculating the instantaneous error 
            En = 0.5*(yn-sn)**2

            # calculating the delta w
            ded = float((neta)*(yn-sn))

            xn = xn*ded
            xn = list(xn)
            xn.insert(0,ded)
      
           # updating parameters
            for k in range(len(w)):
                w[k] = w[k] + xn[k]
            bias = bias + ded
            avg_error += En
        # calculating average error 
        avg_error = float(avg_error/float(n))
        epoch_error.append(avg_error)
        # calculating the delta error for checking convergence 
        delta_error = float(last_error - avg_error)
        if (last_error !=1 and (delta_error<= threshold) and (delta_error>0)):
            converged = True
        last_error = avg_error
    # plotting 
    plt.title("Average Error vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Average Squared Error (En)")
    plt.plot(np.arange(1,len(epoch_error)+1), epoch_error)
    plt.show()
    return w

# estimating the w parameters 
w = perceptron(X_train, y_train)

# testing

y_train_predict = []

# calculating the neuron output
for index in range(len(X_train)):
    output = 0
    xn = X_train.iloc[index]
    xn = list(xn)
    xn.insert(0,1)
    for i in range(len(w)):
        output = output + xn[i]*w[i]
    y_train_predict.append(output)

y_test_predict = []
# predicting the values 
for index in range(len(X_test)):
    output = 0
    xn = X_test.iloc[index]
    xn = list(xn)
    xn.insert(0,1)
    for i in range(len(w)):
        output = output + xn[i]*w[i]
    y_test_predict.append(output)

#%%
# printing the root mean squred error 
print("RMSE of train data is",np.sqrt(mean_squared_error(y_train,y_train_predict)))
print("RMSE of test data is",np.sqrt(mean_squared_error(y_test,y_test_predict)))

#plotting the target values and model values as per requirements
plt.scatter(y_train,y_train_predict)
plt.ylabel("Model Output")
plt.xlabel("Target Output")
plt.show()
plt.scatter(y_test,y_test_predict)
plt.ylabel("Model Output")
plt.xlabel("Target Output")
plt.show()



#%%
# reading the data
dataset1 = pd.read_csv("Regression/BivariateData/22.csv", names = ["X1","X2", "Y"])
X_train1, X_test1, y_train1, y_test1 = train_test_split(dataset1.iloc[:,:-1],dataset1.iloc[:,-1] , test_size=0.3, shuffle=False)
# plt.scatter(X_train1,X_train1)
# plt.show()

# %%
# calculating the w parameteres
w1 = perceptron(X_train1, y_train1)

y_train_predict1 = []
# calculating the neuron output 
for index in range(len(X_train1)):
    output = 0
    xn = X_train1.iloc[index]
    xn = list(xn)
    xn.insert(0,1)
    for i in range(len(w)):
        output = output + xn[i]*w[i]
    y_train_predict1.append(output)

y_test_predict1 = []
# predicting the values
for index in range(len(X_test1)):
    output = 0
    xn = X_test1.iloc[index]
    xn = list(xn)
    xn.insert(0,1)
    for i in range(len(w)):
        output = output + xn[i]*w[i]
    y_test_predict1.append(output)

#%%

# printing the RMSE values 
print("RMSE of train data is",np.sqrt(mean_squared_error(y_train1,y_train_predict1)))
print("RMSE of test data is",np.sqrt(mean_squared_error(y_test1,y_test_predict1)))

# plotting the model output and traget values 
plt.scatter(y_train1,y_train_predict1)
plt.ylabel("Model Output")
plt.xlabel("Target Output")
plt.show()
plt.scatter(y_test1,y_test_predict1)
plt.ylabel("Model Output")
plt.xlabel("Target Output")
plt.show()


# %%
