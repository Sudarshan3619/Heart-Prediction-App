from flask import *
import pickle
import numpy as np
from scaling import data_preprocessing


app=Flask(__name__)


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
        l=data_preprocessing(data)
        inputs=np.array([l])
        #print(inputs)
        #model=pickle.load(open('heart.pkl','rb'))
        result=rfc.predict(inputs)
        return render_template('output.html',data=result[0])

@app.route('/fourmodelclassifier')
def fourmodelclassifier():
    return render_template('input1.html')

@app.route('/predict1',methods=['POST'])
def predict1():
    if request.method=='POST':
        data=request.form
        l=data_preprocessing(data)
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
    with open('heart_all.pkl','rb') as f:
        rfc,lr,abc,knn=pickle.load(f)
    app.run(debug=True)
