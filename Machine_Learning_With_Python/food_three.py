import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

'''
Visualizing a Decision Tree
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

model = DecisionTreeClassifier()
model.fit(x, y)

tree.export_graphviz(model, out_file='dessert-recommender.dot',
                     feature_names=['Age'],
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)
