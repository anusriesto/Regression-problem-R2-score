import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#now importing the data
dt=pd.read_csv(r'C:\Users\LENOVO\Desktop\50_Startups.csv')
dt.head(5)
x=dt.iloc[:,:-1]
y=dt.iloc[: ,4 ]
states=pd.get_dummies(x['State'],drop_first=True)
x=x.drop('State',axis=1)
x=pd.concat([x,states],axis=1)
#now spliting of test and train data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=3)


##training the model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

#now prediction of the model
y_pred=lr.predict(x_test)




#now for the sake of comparison we are using r2 method
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)



