from flask import *
import pickle
import numpy as np
from scaling import scaling
import os



app=Flask(__name__)

#model=pickle.load(open('heart.pkl','rb'))
#scaling=pickle.load(open('scaling_file.pkl','rb'))

with open('heart_all.pkl','rb') as f:
    rfc,lr,abc,knn,scale=pickle.load(f)

def scaler(x,mean,sd):
    return ((x-mean)/sd)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/randomforestclassifier')
def randforest():
    return render_template('input.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        data=request.form

        l=[]   
        l.append(int(data['male']))
        l.append(int(data['age']))
        l.append(float(data['cigsPerDay']))
        l.append(int(data['prevalentStroke']))
        l.append(int(data['prevalentHyp']))
        l.append(int(data['diabetes']))
        
        a=float(data['sysBP'])
        if a>0:
            a=np.log(a)
        l.append(a)
        i=0
        for feature in ['male','age','cigsPerDay','prevalentStroke','prevalentHyp','diabetes','sysBP']:
            l[i]=scaler(l[i],scale.mean(feature),scale.sd(feature))
            i+=1;
        
        inputs=np.array([l])
        #print(inputs)
        #model=pickle.load(open('heart.pkl','rb'))
        result=rfc.predict(inputs)
        return render_template('output.html',data=result[0])

@app.route('/fourmodelclassifier')
def fourmodelclassifier():
    return render_template('input1.html')

@app.route('/predict1',methods=['POST'])
def predict_by4():
    if request.method=='POST':
        data=request.form
        l=[]
        l.append(int(data['male']))
        l.append(int(data['age']))
        l.append(float(data['cigsPerDay']))
        l.append(int(data['prevalentStroke']))
        l.append(int(data['prevalentHyp']))
        l.append(int(data['diabetes']))
        
        a=float(data['sysBP'])
        
        #l.append(a)
        if a>0:
            a=np.log(a)
        
        l.append(a)
        i=0
        for feature in ['male','age','cigsPerDay','prevalentStroke','prevalentHyp','diabetes','sysBP']:
            l[i]=scaler(l[i],scale.mean(feature),scale.sd(feature))
            i+=1;
        
        #inputs=np.array([l])
        
        inputs=np.array([l])
        result=[0,0,0,0]
        #model=pickle.load(open('heart.pkl','rb'))
        result[0]=rfc.predict(inputs)[0]
        result[1]=lr.predict(inputs)[0]
        result[2]=abc.predict(inputs)[0]
        result[3]=knn.predict(inputs)[0]
        if sum(result)>=2:
            result[0]=1
        else:
            result[0]=0
            
        return render_template('output.html',data=result[0])    
        
        
if __name__=='__main__':
    app.run(debug=True)
