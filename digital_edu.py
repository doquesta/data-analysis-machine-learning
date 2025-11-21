#create your individual project here!
import pandas as pd 
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load data
df = pd.read_csv('train.csv')

# 2. Convert 'bdate' to age
def calculate_age(bdate):
    try:
        year = int(str(bdate).split('.')[-1])
        return datetime.now().year - year if year > 1900 else np.nan
    except:
        return np.nan

df['age'] = df['bdate'].apply(calculate_age)
df['age'].fillna(df['age'].median(), inplace=True)

# 3. Drop unused columns
df.drop(columns=['bdate', 'langs', 'last_seen', 'occupation_name', 'd'], inplace=True, errors='ignore')

# 4. Fill missing values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include='object').columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 5. Convert boolean strings to integers
for col in ['has_photo', 'has_mobile', 'life_main', 'people_main']:
    df[col] = df[col].astype(str).str.lower().replace({'true': 1, 'false': 0})
    df[col] = df[col].astype(float).fillna(0).astype(int)
# 6. Encode categorial columns
label_cols = ['sex', 'education_form', 'education_status', 'city', 'occupation_type']
for col in label_cols:
    df[col] = df[col].astype(str) # konversi aman ke string dulu
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

#7. Select features and target
features = [
    'sex', 'age', 'has_photo', 'has_mobile', 'followers_count', 'graduation',
    'education_form', 'relation', 'education_status', "life_main",
    'people_main', 'city', 'occupation_type', 'career-start', 'career_end'
]
df.info()


