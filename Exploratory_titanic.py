import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv('titanic_dataset.csv')
print(df.info())
print(df.head())
print(df.isnull().sum())
df.drop("Cabin",axis=1,inplace=True)
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
print(df.isnull().sum())

#Outlier Detection and Removal
df.boxplot()
data=df.copy()
df=df.drop("Name", axis=1)
df=df.drop("Sex", axis=1)
df=df.drop("Ticket", axis=1)
df=df.drop("Embarked", axis=1)

from scipy import stats
z=np.abs(stats.zscore(df))
df2=df.copy()
q1=df2.quantile(0.25)
q3=df2.quantile(0.75)
IQR=q3-q1
df2_new=df2[((df2>=q1-1.5*IQR)&(df2<=q3+1.5*IQR)).all(axis=1)]
df2_new.boxplot()



#STATISTICAL METHOD OF VIEWING THE DATA
print(data["Embarked"].value_counts())
print(data["Pclass"].value_counts())
print(data["Survived"].value_counts())
print(pd.crosstab(data["Pclass"],data["Survived"]))
print(pd.crosstab(data["Sex"],data["Survived"]))

#Graphical Method of displaying data
print(sns.countplot(x='Survived',data=data))
print(sns.countplot(x='Pclass',data=data))
print(sns.countplot(x='Sex',data=data))
print(sns.displot(data["Age"]))
print(sns.displot(data["Fare"]))
print(sns.countplot(x='Pclass',hue='Survived',data=data))
print(sns.countplot(x='Age',hue='Survived',data=data))
print(sns.countplot(x='Sex',hue='Survived',data=data))
print(sns.displot(data[data['Survived']==0]["Age"]))
print(sns.displot(data[data['Survived']==1]["Age"]))
(sns.heatmap(data.corr(),annot=True))
