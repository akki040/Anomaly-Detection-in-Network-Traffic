import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Load Titanic dataset
df = sns.load_dataset('titanic')
df = df[['sex', 'age', 'fare', 'class', 'embarked', 'survived']].dropna()

# Encode categorical features
le_sex = LabelEncoder()
le_class = LabelEncoder()
le_embarked = LabelEncoder()

df['sex_enc'] = le_sex.fit_transform(df['sex'])
df['class_enc'] = le_class.fit_transform(df['class'])
df['embarked_enc'] = le_embarked.fit_transform(df['embarked'])

# Features and label
X = df[['sex_enc', 'age', 'fare', 'class_enc', 'embarked_enc']]
y = df['survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Set default save directory
save_dir = r"C:\Users\akash\Desktop\New folder\1celebal\Assignment 7"

# Save model
model_path = os.path.join(save_dir, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# Save encoders
encoder_path = os.path.join(save_dir, "encoders.pkl")
with open(encoder_path, "wb") as f:
    pickle.dump((le_sex, le_class, le_embarked), f)

print(f"✅ Model saved to: {model_path}")
print(f"✅ Encoders saved to: {encoder_path}")
