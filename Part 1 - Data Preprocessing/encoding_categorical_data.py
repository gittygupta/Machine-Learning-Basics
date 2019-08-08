from sklearn.preprocessing import LabelEncoder
labelencoder_country = LabelEncoder()
x[:, 0] = labelencoder_country.fit_transform(x[:, 0])

# Dummy encoding
m = pd.get_dummies(pd.Series(x[:, 0]))
data = pd.DataFrame({'Country':x[:,0],'Age':x[:,1], 'Salary':x[:,2]})
data = data.drop('Country', axis = 1)
data = pd.concat([m, data], axis=1)
x = data.iloc[:, :].values
