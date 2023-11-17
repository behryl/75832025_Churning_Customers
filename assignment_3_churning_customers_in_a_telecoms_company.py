# -*- coding: utf-8 -*-
"""Assignment 3: Churning Customers in a Telecoms Company

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vqeMxC5r1Fc6JCZTMSrATt5A8-jDwMV8
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pickle as pkl

from google.colab import drive

drive.mount("/content/drive")

df = os.path.abspath("/content/drive/MyDrive/CustomerChurn_dataset.csv")

churningcustomer = pd.read_csv(df)

churningcustomer

churningcustomer = churningcustomer.drop('customerID', axis=1)

churningcustomer

churningcustomer.info()

churningcustomer['TotalCharges'] = pd.to_numeric(churningcustomer['TotalCharges'], errors='coerce')



print(churningcustomer.isnull().sum())

imputer = SimpleImputer(strategy='mean')
churningcustomer['TotalCharges'] = imputer.fit_transform(churningcustomer[['TotalCharges']])

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Assuming 'Churn' is your target variable
X = churningcustomer.drop('Churn', axis=1)  # Features
y = churningcustomer['Churn']  # Target variable

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Apply label encoding to categorical columns
label_encoder = LabelEncoder()
X_encoded = X.copy()  # Make a copy to avoid modifying the original DataFrame
for col in categorical_cols:
    X_encoded[col] = label_encoder.fit_transform(X[col])

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Fit the model to your data
rf_classifier.fit(X_encoded, y)

# Get feature importances from the trained model
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)

# Display the top 5 features by importance
top_features = feature_importances.head(5)
print("Top Features by Importance:")
print(top_features)



"""EDA"""

top_features = ['TotalCharges', 'MonthlyCharges', 'tenure', 'Contract', 'PaymentMethod']
print(churningcustomer[top_features].describe())

for feature in top_features:
    sns.histplot(churningcustomer[feature], kde=True)
    plt.title(f'Distribution of {feature}')

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')

    plt.show()

for feature in ['TotalCharges', 'MonthlyCharges', 'tenure']:
    sns.boxplot(x='Churn', y=feature, data=churningcustomer)
    plt.title(f'Boxplot of {feature} by Churn')
    plt.show()

for feature in ['Contract', 'PaymentMethod']:
    sns.countplot(x=feature, hue='Churn', data=churningcustomer)
    plt.title(f'Distribution of {feature} by Churn')

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')

    plt.show()

top_features_matrix = churningcustomer[top_features].corr()
sns.heatmap(top_features_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Top Features')
plt.show()

"""Training Model"""

# After label encoding
print(X_encoded.dtypes)

df = pd.concat([churningcustomer[top_features], y], axis=1)



categorical_cols = df.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

df

X = df. drop('Churn', axis=1)
y = df['Churn']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X.copy())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# Keras Functional API model

#Input layer
input_layer = Input(shape=(X_train.shape[1],))

#Hidden layer
hidden_layer_1 = Dense(32, activation='relu')(input_layer)
hidden_layer_2 = Dense(24, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(12, activation='relu')(hidden_layer_2)


#)Outputvlayer
output_layer = Dense(1, activation='sigmoid')(hidden_layer_3)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=120, batch_size=32, validation_data=(X_test, y_test))

_, accuracy = model.evaluate(X_train, y_train)
accuracy*100

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.4f}')

!pip install keras-tuner

import keras_tuner
from tensorflow import keras

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(X_train.shape[1],)))

    # Tune the number of hidden layers and units
    for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=4)):
        model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=96, step=32),
                             activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh'])))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Tune the learning rate
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    return model

from google.colab import drive
drive.mount('/content/drive')

build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.Hyperband(
  hypermodel=build_model,
  objective='val_accuracy',
  max_epochs=100,
  factor=3,
  directory='tuning_dir',
  project_name='samples')

tuner.search(X_train, y_train, epochs=30 ,validation_data=(X_test, y_test))

tuner.results_summary()

best_model = tuner.get_best_models(num_models=2)[0]

best_model.summary()

test_accuracy = best_model.evaluate(X_test, y_test)[1]
print(f"Test Accuracy: {test_accuracy:.4f}")



#Deployment
best_model.save("model.h5")

with open ('sc.pkl','wb') as file:
  pkl.dump(scaler,file)

with open('lbl_encoder.pkl', 'wb') as file:
  pkl.dump(label_encoder,file)