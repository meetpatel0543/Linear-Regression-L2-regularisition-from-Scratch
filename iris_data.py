import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

#loading iris dataset 
iris = load_iris()

#Splitting the data into 10% test and 90% train
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=42,stratify=iris.target)

sepal_len_X_train = X_train[:,0]
sepal_width_X_train = X_train[:,1]
petal_len_X_train = X_train[:,2]
petal_wid_X_train = X_train[:,3]
sepal_len_test = X_test[:,0]
sepal_width_X_test = X_test[:,1]
petal_len_X_test = X_test[:,2]
petal_wid_X_test = X_test[:,3]

# 1 Sepal Length vs Sepal Width without regularization
model1 = LinearRegression()
model1.fit(X=sepal_len_X_train,y=sepal_width_X_train,batch_size=32,max_epochs=100,alpha=0.0005,regularization=0,patience = 3,problem=1)
plt.plot(model1.train_error_list)
plt.plot(model1.validation_error_list)
plt.title('sepal length vs sepal width without regularization')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(["traininig loss", "vlaidation loss"])
plt.show()
predict1 = model1.predict(sepal_len_X_train)
model1_acc = model1.score(predict1,sepal_width_X_train)
print("sepal length vs sepal width without regularization accuracy ", model1_acc)
print("weights ",model1.weights)
print("--------------------------------------")
# 1 Sepal Length vs Sepal Width with regularization
model1 = LinearRegression()
model1.fit(X=sepal_len_X_train,y=sepal_width_X_train,batch_size=32,max_epochs=100,alpha=0.0005,regularization=0.1,patience = 3,problem=1)
plt.plot(model1.train_error_list)
plt.plot(model1.validation_error_list)
plt.title('sepal length vs sepal width with regularization')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(["traininig loss", "vlaidation loss"])
plt.show()
predict1 = model1.predict(sepal_len_X_train)
model1_acc = model1.score(predict1,sepal_width_X_train)
print("sepal length vs sepal width with regularization accuracy ", model1_acc)
print("weights ",model1.weights)
print("--------------------------------------")

# 2 Sepal Lenght vs Petal Length without regularization
model1 = LinearRegression()
model1.fit(sepal_len_X_train,petal_len_X_train,batch_size=32,max_epochs=100,alpha=0.0005,regularization=0,patience = 3,problem=1)
plt.plot(model1.train_error_list)
plt.plot(model1.validation_error_list)
plt.title('Sepal Lenght vs Petal Length without regularization')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(["traininig loss", "vlaidation loss"])
plt.show()

predict1 = model1.predict(sepal_len_X_train)
model1_acc = model1.score(predict1,sepal_width_X_train)
print("Sepal Lenght vs Petal Length without regularization accuracy ", model1_acc)
print("weights ",model1.weights)
print("--------------------------------------")

# 2 Sepal Lenght vs Petal Length with regularization
model1 = LinearRegression()
model1.fit(sepal_len_X_train,petal_len_X_train,batch_size=32,max_epochs=100,alpha=0.0005,regularization=0.1,patience = 3,problem=1)
plt.plot(model1.train_error_list)
plt.plot(model1.validation_error_list)
plt.title('Sepal Lenght vs Petal Length with regularization')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(["traininig loss", "vlaidation loss"])
plt.show()

predict1 = model1.predict(sepal_len_X_train)
model1_acc = model1.score(predict1,sepal_width_X_train)
print("Sepal Lenght vs Petal Length with regularization accuracy ", model1_acc)
print("weights ",model1.weights)
print("--------------------------------------")

# 3 Sepal Lenght vs Petal Width without regularization
model1 = LinearRegression()
model1.fit(sepal_len_X_train,petal_wid_X_train,batch_size=32,max_epochs=100,alpha=0.0005,regularization=0,patience = 3,problem=1)
plt.plot(model1.train_error_list)
plt.plot(model1.validation_error_list)
plt.title('Sepal Lenght vs Petal Width without regularization')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(["traininig loss", "vlaidation loss"])
plt.show()

predict1 = model1.predict(sepal_len_X_train)
model1_acc = model1.score(predict1,sepal_width_X_train)
print("Sepal Lenght vs Petal Width without regularization accuracy ", model1_acc)
print("weights ",model1.weights)
print("--------------------------------------")

# 3 Sepal Lenght vs Petal Width with regularization
model1 = LinearRegression()
model1.fit(sepal_len_X_train,petal_wid_X_train,batch_size=32,max_epochs=100,alpha=0.0005,regularization=0.1,patience = 3,problem=1)
plt.plot(model1.train_error_list)
plt.plot(model1.validation_error_list)
plt.title('Sepal Lenght vs Petal Width with regularization')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(["traininig loss", "vlaidation loss"])
plt.show()

predict1 = model1.predict(sepal_len_X_train)
model1_acc = model1.score(predict1,sepal_width_X_train)
print("Sepal Lenght vs Petal Width with regularization accuracy ", model1_acc)
print("weights ",model1.weights)
print("--------------------------------------")

#4 Sepal Width vs Petal Width without regularisation
model1 = LinearRegression()
model1.fit(sepal_width_X_train,petal_wid_X_train,batch_size=32,max_epochs=100,alpha=0.0005,regularization=0,patience = 3,problem=1)
plt.plot(model1.train_error_list)
plt.plot(model1.validation_error_list)
plt.title('Sepal Width vs Petal Width without regularisation')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(["traininig loss", "vlaidation loss"])
plt.show()

predict1 = model1.predict(sepal_len_X_train)
model1_acc = model1.score(predict1,sepal_width_X_train)
print("Sepal Width vs Petal Width without regularisation accuracy ", model1_acc)
print("weights ",model1.weights)
print("--------------------------------------")

#4 Sepal Width vs Petal Width with regularisation
model1 = LinearRegression()
model1.fit(sepal_width_X_train,petal_wid_X_train,batch_size=32,max_epochs=100,alpha=0.0005,regularization=0.1,patience = 3,problem=1)
plt.plot(model1.train_error_list)
plt.plot(model1.validation_error_list)
plt.title('Sepal Width vs Petal Width with regularisation')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend(["traininig loss", "vlaidation loss"])
plt.show()

predict1 = model1.predict(sepal_len_X_train)
model1_acc = model1.score(predict1,sepal_width_X_train)
print("Sepal Width vs Petal Width with regularisation accuracy ", model1_acc)
print("weights ",model1.weights)
print("--------------------------------------")
