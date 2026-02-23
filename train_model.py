# =============================================
# AI-Powered Smart Career Path Predictor
# train_model.py
# =============================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("Training Started...")

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
try:
    df = pd.read_csv("career_recommender.csv")
    print("Dataset Loaded Successfully")
except FileNotFoundError:
    print("ERROR: career_recommender.csv not found in this folder")
    exit()

# Remove extra spaces in column names
df.columns = df.columns.str.strip()

# -----------------------------
# ⭐ IMPORTANT: Create Career Column
# -----------------------------
for col in df.columns:
    if "Job title" in col:
        df.rename(columns={col: "Career"}, inplace=True)

print("Updated Columns:")
print(df.columns)

# -----------------------------
# Step 2: Drop Unnecessary Columns
# -----------------------------
if "What is your name?" in df.columns:
    df.drop(["What is your name?"], axis=1, inplace=True)
    print("Dropped Name Column")

# -----------------------------
# Step 3: Handle Missing Values
# -----------------------------
df.fillna("Unknown", inplace=True)
print("Missing Values Filled")
# Remove rare careers (less than 5 samples)
career_counts = df["Career"].value_counts()

# Keep only careers that appear at least 5 times
valid_careers = career_counts[career_counts >= 5].index

df = df[df["Career"].isin(valid_careers)]

print("After Removing Rare Careers:")
print("Total Unique Careers:", df["Career"].nunique())
print(df["Career"].value_counts())

# -----------------------------
# Step 4: Remove NA Careers
# -----------------------------
df = df[df["Career"] != "NA"]
df = df[df["Career"] != "Unknown"]
df = df[df["Career"] != "NA"]
df = df[df["Career"] != "Unknown"]
# -----------------------------
# Step 4B: Group Similar Careers (IMPORTANT)
# -----------------------------

# Convert to lowercase
df["Career"] = df["Career"].str.lower()

def group_career(title):
    
    if "student" in title:
        return "Student"
    
    elif "software" in title or "developer" in title or "programmer" in title:
        return "Software Engineer"
    
    elif "data" in title or "analyst" in title:
        return "Data Science"
    
    elif "mechanical" in title or "civil" in title or "electrical" in title:
        return "Core Engineering"
    
    elif "teacher" in title or "lecturer" in title or "professor" in title:
        return "Education"
    
    elif "account" in title or "finance" in title or "bank" in title:
        return "Finance"
    
    elif "manager" in title or "business" in title:
        return "Management"
    
    elif "sales" in title or "marketing" in title:
        return "Sales & Marketing"
    
    else:
        return "Other"

df["Career"] = df["Career"].apply(group_career)

print("New Unique Careers:", df["Career"].nunique())
print(df["Career"].value_counts())

print("Total Unique Careers:", df["Career"].nunique())
print(df["Career"].value_counts())


# -----------------------------
# Step 5: NLP Vectorization
# -----------------------------
print("Applying NLP Vectorization...")

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(
    df["What are your skills ? (Select multiple if necessary)"]
)

# -----------------------------
# Step 6: Encode Career Column
# -----------------------------
career_encoder = LabelEncoder()
y = career_encoder.fit_transform(df["Career"])

# -----------------------------
# Step 7: Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", model.score(X_test, y_test))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = model.score(X_test, y_test)
print(f"\nModel Trained Successfully with Accuracy: {accuracy*100:.2f}%")
# Confusion Matrix
cm = confusion_matrix(y_test, model.predict(X_test))

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# -----------------------------
# Step 8: Save Model
# -----------------------------
pickle.dump(model, open("career_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(career_encoder, open("career_encoder.pkl", "wb"))

print("Model, vectorizer, and encoder saved successfully ✅")