import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("Merged_data.csv")

# Placeholder MSRP during training
neutral_msrp = df['MSRP'].median()
df['MSRP'] = neutral_msrp

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
    threshold = df['value_score'].quantile(0.6)
    df['is_good_deal'] = (df['value_score'] >= threshold).astype(int)
    return df

labeled_df = generate_target(df, persona='family')

# One-hot encode
encoded_df = pd.get_dummies(
    labeled_df.drop(columns=['value_score']),
    columns=['Vehicle Size', 'Vehicle Style'],
    drop_first=True
)

X = encoded_df.drop(columns=['is_good_deal', 'Make', 'Model', 'Year', 'Transmission Type',
                             'Driven_Wheels', 'Market Category'], errors='ignore')
y = encoded_df['is_good_deal']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('trained_model.sav', 'wb') as f:
    pickle.dump(model, f)
