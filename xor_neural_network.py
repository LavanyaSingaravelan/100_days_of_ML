#2 layer neural network

import numpy as np
import time


#variables
n_hidden=10
n_in=10
#outputs
n_out=10
#sample data
n_sample=300

#hyperparameters
learning_rate=0.01
momentum=0.9

#non deterministic seeding
np.random.seed(0)

#activation function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def tanh_prime(x):
    return 1-np.tanh(x)**2


#training function
#x-input data
#t-transpose
#V-layer1
#W-layer2
#bv,bw-biases 
def train(x,t,V,W,bv,bw):
    #forward --matrix multiply+biases
    A=np.dot(x,V)+bv #hidden layer input=matrix_dot_product(x,v)+bias(bv)
    Z=np.tanh(A)

    B=np.dot(Z,W)+bw#take the output of the first layer and put it in the second one
    Y=sigmoid(B)
    
    #backward
    Ew=Y-t
    Ev=tanh_prime(A)*np.dot(W,Ew)

   #predict our loss
    dW=np.outer(Z,Ew)
    dV=np.outer(x,Ev)

#cross entropy because of classification 
    loss=-np.mean(t*np.log(Y)+(1-t)*np.log(1-Y))
    return loss,(dV,dW,Ev,Ew)




#prediction function
def predict(x,V,W,bv,bw):
    A=np.dot(x,V)+bv
    B=np.dot(np.tanh(A),W)+bw
    return (sigmoid(B)>0.5).astype(int)

#create layers
V=np.random.normal(scale=0.1,size=(n_in,n_hidden))
W=np.random.normal(scale=0.1,size=(n_hidden,n_out))

#creating biases
bv=np.zeros(n_hidden)
bw=np.zeros(n_out)

params=[V,W,bv,bw] #easy to input as an array into the training function

#generate our data
X=np.random.binomial(1,0.5,(n_sample,n_in))#n_samples=>generating 300 samples
T=X^1

#training time
for epoch in range(100):
    err=[]
    upd=[0]*len(params)

    t0=time.clock()
    #for each data point we are going to update the weights of our network
    for i in range(X.shape[0]):
        loss,grad=train(X[i],T[i],*params)
        #update los
        for j in range(len(params)):
            params[j]-=upd[j]

        for j in range(len(params)):
            upd[j]=learning_rate*grad[j]+momentum*upd[j]#calculate the loss
            
        err.append(loss)

    print('Epoch:%d,Loss:%.8f,Time:%.4fs'%(epoch,np.mean(err),time.clock()-t0))


#try to predict something          
x=np.random.binomial(1,0.5,n_in)
print('XOR prediction')
print(x)
print(predict(x,*params))

























    

    
    
