import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import altair as alt

# Load dataset
data_path = '/Users/omarallam/Downloads/breast-cancer.csv'
data = pd.read_csv(data_path)

# Map diagnosis to binary values: M -> 0, B -> 1
data['target'] = data['diagnosis'].map({'M': 0, 'B': 1})

# Drop unnecessary columns
df = data.drop(['id', 'diagnosis'], axis=1)

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title('Breast Cancer Prediction using Decision Tree')

# Description of the model and dataset
st.header('Model and Dataset Description')
st.write("""
The Breast Cancer Wisconsin dataset is used for training a decision tree classifier to predict whether a tumor is malignant or benign.
The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast mass. The target variable is a binary classification indicating whether the tumor is malignant (encoded as 0) or benign (encoded as 1).

### Features:
- Mean radius, texture, perimeter, area, smoothness, etc.
- Mean, standard error, and worst (mean of the three largest values) of these features.

### Model:
A Decision Tree Classifier is trained on the dataset to predict the target variable. The accuracy of the model on the test set is displayed below.
""")
st.write(f'Model Accuracy: {accuracy:.2f}')

# Feature importance visualization
st.header('Feature Importance')
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
chart = alt.Chart(feature_importance).mark_bar().encode(
    x='Importance',
    y=alt.Y('Feature', sort='-x')
)
st.altair_chart(chart, use_container_width=True)

# Prediction interface
st.header('Predict Breast Cancer')
input_data = []
for feature in X.columns:
    value = st.text_input(f'{feature}', '0')
    input_data.append(float(value))

if st.button('Predict'):
    prediction = model.predict([input_data])
    prediction_proba = model.predict_proba([input_data])
    result = 'Malignant' if prediction[0] == 0 else 'Benign'
    st.write(f'Prediction: {result}')
    st.write(f'Prediction Probability: {prediction_proba[0]}')
