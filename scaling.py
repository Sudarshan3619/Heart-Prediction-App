import pandas as pd
import numpy as np 
def scaler(x,feature):
    data=pd.read_csv("Data.csv")
    mean=data[feature].mean()
    sd=data[feature].std()
    return ((x-mean)/sd)

## Data Pre-Processing 
def data_preprocessing(data):
    l=[];
    for feature,_ in data.items():
        l.append(float(data[feature]))
    
    a=float(data['sysBP'])
    if a>0:
        a=np.log(a)
    l[-1]=a;
    i=0
    for feature in ['male','age','cigsPerDay','prevalentStroke','prevalentHyp','diabetes','sysBP']:
        l[i]=scaler(l[i],feature)
        i+=1
    return l;

        


    