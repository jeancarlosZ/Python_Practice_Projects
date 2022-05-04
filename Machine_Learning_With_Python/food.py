import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
Train Model and Calculate Accuracy
'''
food_data = pd.read_csv('Food_Preference.csv')
x = food_data.drop(columns=['Timestamp',
                            'Participant_ID',
                            'Gender',
                            'Nationality',
                            'Food',
                            'Juice',
                            'Dessert'])
y = food_data['Dessert']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

score = accuracy_score(y_test, predictions)

print(score)
