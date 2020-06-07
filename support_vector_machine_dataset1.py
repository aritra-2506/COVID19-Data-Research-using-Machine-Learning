import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Data reading
data = pd.read_csv("COVID-19.csv")

#Data preprocessing

labelencoder = LabelEncoder()
onehotencoder=OneHotEncoder()

data=data.iloc[:,1:]
data.iloc[:, 3] = labelencoder.fit_transform(data.iloc[:, 3])

x=data.iloc[:,1]
data=data.drop('Diagnosis', axis=1)
data['Diagnosis'] = x

j=len(data.columns)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[j-1])], remainder='passthrough')
data.iloc[:, j-1] = labelencoder.fit_transform(data.iloc[:, j-1])

diagnosis=pd.DataFrame(onehotencoder.fit_transform(data[['Diagnosis']]).toarray())
diagnosis=diagnosis.iloc[:,0]

data=data.drop('Diagnosis', axis=1)
data=data.join(diagnosis)
data=data.rename(columns={0: "Diagnosis"})

#Train-Test split
data_train=data[0:146]
data_test=data[146:179]

data_train=data_train.drop('Dataset', axis=1)
data_test=data_test.drop('Dataset', axis=1)

x_train=data_train.iloc[:,:-1]
y_train=data_train.iloc[:,-1]

x_test=data_test.iloc[:,:-1]
y_test=data_test.iloc[:,-1]

model = SVC(kernel='linear')

#Fitting
model.fit(x_train, y_train)

# Making the Confusion Matrix

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
