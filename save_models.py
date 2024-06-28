import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # or the specific module you need

# Assuming you have your training data and model
# Replace these with your actual training data and model
X_train = ...  # Your training data
y_train = ...  # Your training labels

# Train the models
soundrep_model = RandomForestClassifier()
soundrep_model.fit(X_train, y_train)

wordrep_model = RandomForestClassifier()
wordrep_model.fit(X_train, y_train)

prolongation_model = RandomForestClassifier()
prolongation_model.fit(X_train, y_train)

# Save the models
with open('soundrep_model.pkl', 'wb') as f:
    pickle.dump(soundrep_model, f)

with open('wordrep_model.pkl', 'wb') as f:
    pickle.dump(wordrep_model, f)

with open('prolongation_model.pkl', 'wb') as f:
    pickle.dump(prolongation_model, f)