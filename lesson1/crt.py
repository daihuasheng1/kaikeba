from sklearn.model_selection   import train_test_split
from sklearn import  preprocessing
from sklearn.metrics  import accuracy_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeRegressor


digits = load_digits()
data = digits.data
train_x,test_x,train_y,test_y = train_test_split(data,digits.target,test_size=0.15,random_state=11)

ss = preprocessing.StandardScaler()
train_ss_x =  ss.fit_transform(train_x)
test_ss_x = ss.fit_transform(test_x)

crt = DecisionTreeRegressor()
crt.fit(train_ss_x,train_y)
predict_y = crt.predict(test_ss_x)
print(accuracy_score(predict_y,test_y))