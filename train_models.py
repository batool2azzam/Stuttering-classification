import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load your training data
# Replace 'your_training_data.csv' with the path to your actual training data
data = pd.read_csv('your_training_data.csv')
X_train = data.drop('target_column', axis=1)  # Replace 'target_column' with the name of your target column
y_train = data['target_column']

# Train your models
soundrep_model = RandomForestClassifier()
soundrep_model.fit(X_train, y_train)

wordrep_model = RandomForestClassifier()
wordrep_model.fit(X_train, y_train)

prolongation_model = RandomForestClassifier()
prolongation_model.fit(X_train, y_train)

# Save the models
with open('models/soundrep_model.pkl', 'wb') as f:
    pickle.dump(soundrep_model, f)

with open('models/wordrep_model.pkl', 'wb') as f:
    pickle.dump(wordrep_model, f)

with open('models/prolongation_model.pkl', 'wb') as f:
    pickle.dump(prolongation_model, f)

print("Models have been trained and saved successfully.")
