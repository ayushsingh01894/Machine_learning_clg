'''from sklearn.linear_model import LinearRegression
X = [[1],[2],[3],[4]]
y = [40,50,60,70]
model = LinearRegression()
model.fit(X,y)
hours = float(input("Enter how many hours you studies="))
predicted_marks = model.predict([[hours]])
print(f"Based on your hour{hours} scored is {predicted_marks}")'''



#logistic regression
'''from sklearn.linear_model import LogisticRegression
X = [[1],[2],[3],[4],[5]]
y = [0,0,1,1,1]

model = LogisticRegression()
model.fit(X,y)
hours = float(input("Enter how many hours you studies="))
result = model.predict([[hours]])[0]
if result == 1:
    print(f"based on hour{hours}, you are likely to PASS")
else:
    print(f"based on hour{hours}, you are likely to Fail")'''

'''
#KNN regression

from sklearn.neighbors import KNeighborsClassifier
X = [
    [180,7],
    [200,7.5],
    [250,8],
    [300,8.5],
    [330,90],
    [360,9.5]
]
y = [0,0,0,1,1,1]
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)
weight = float(input("Enter the weight in grams:"))
size = float(input("Enter the size in cm:"))
prediction = model.predict([[weight , size]])[0]
if prediction == 1:
    print("this is likely an apple")
else:
    print("this is likely an orange")'''

'''
#decision tree 
from sklearn.tree import DecisionTreeClassifier
X = [
    [7,2],
    [8,3],
    [9,8],
    [10,9]
]
y = [0,0,1,1]
model = DecisionTreeClassifier()
model.fit(X,y)
size = float(input("Enter the fruit size in cm:"))
shade = float(input("Enter the color shade(1-10):"))

result = model.predict([[size,shade]])[0]
if result == 1:
    print("this is likely an apple")
else:
    print("this is likely an orange")'''
