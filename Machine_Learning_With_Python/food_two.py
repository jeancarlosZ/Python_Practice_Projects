import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

'''
Persist Model
'''
# food_data = pd.read_csv('Food_Preference.csv')
# x = food_data.drop(columns=['Timestamp',
#                             'Participant_ID',
#                             'Gender',
#                             'Nationality',
#                             'Food',
#                             'Juice',
#                             'Dessert'])
# y = food_data['Dessert']

# model = DecisionTreeClassifier()
# model.fit(x, y)

# joblib.dump(model, 'dessert-recommender.joblib')


'''
Load Model
'''
model = joblib.load('dessert-recommender.joblib')
predictions = model.predict([[15]])
print(predictions)
