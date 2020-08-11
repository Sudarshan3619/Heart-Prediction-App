class scaling:
    def mean(self,feature):
        data=pd.read_csv("heart prediction.csv")
        return data[feature].mean()
        
    def sd(self,feature):
        data=pd.read_csv("heart prediction.csv")
        return data[feature].std()


import pandas as pd

    