import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

dataset = pd.read_csv('cleaned_car2.csv')

def generate_target(df, persona='family'):
    df = df.copy()

    if persona == 'family':
        df['value_score'] = (
            df['combination_mpg'] * 0.4 +
            df['Number of Doors'].replace({2: 0, 4: 1}).fillna(0.5) * 0.2 +
            df['Vehicle Size'].replace({'Compact': 0.5, 'Midsize': 0.8, 'Large': 1.0}).fillna(0.6) * 0.3 -
            df['MSRP'] * 0.00005
        )

    elif persona == 'sports':
        df['value_score'] = (
            df['Engine HP'] * 0.5 +
            df['displacement'] * 0.3 +
            df['Vehicle Style'].str.lower().str.contains('coupe|convertible').astype(int) * 0.2 -
            df['MSRP'] * 0.00005
        )

    # Label as good deal if in top 40 percent
    threshold = df['value_score'].quantile(0.6)
    df['is_good_deal'] = (df['value_score'] >= threshold).astype(int)

    return df

# as example, apply family buyer persona
labeled_df = generate_target(dataset, persona='family')

# Check label distribution
print(labeled_df['is_good_deal'].value_counts())

encoded_df = pd.get_dummies(
    labeled_df.drop(columns=['value_score']),  # Keep 'is_good_deal'
    columns=['Vehicle Size', 'Vehicle Style'],
    drop_first=True
)

X = encoded_df.drop(columns=['is_good_deal'])
y = encoded_df['is_good_deal']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Saving trained 'model'
filename = 'trained_model.sav'  
pickle.dump(model, open(filename, 'wb'))