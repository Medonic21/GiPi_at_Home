import pandas as pd

# car_features_df = pd.read_csv('C:/Users/Ahmed emad/Desktop/AI Assignment/car_data.csv')

# car_performance_df = pd.read_csv('C:/Users/Ahmed emad/Desktop/AI Assignment/data.csv')

# car_features_df.rename(columns={
#     'make': 'Make', 'model': 'Model', 'year': 'Year'
# }, inplace=True)

# car_performance_df.rename(columns={
#     'make': 'Make', 'model': 'Model', 'year': 'Year'
# }, inplace=True)

# def clean_keys(df):
#     df['Make'] = df['Make'].str.strip().str.lower()
#     df['Model'] = df['Model'].str.strip().str.lower()
#     df['Year'] = df['Year'].astype(str).str.strip()
#     return df

# car_features_df = clean_keys(car_features_df)
# car_performance_df = clean_keys(car_performance_df)

# print("\nSample rows from Car Features dataset:")
# print(car_features_df.head())

# print("\nSample rows from Car Performance dataset:")
# print(car_performance_df.head())

# Car_Dataset = pd.merge(
#     car_features_df,
#     car_performance_df,
#     on=['Make', 'Model', 'Year'],
#     how='inner'
# )
Car_Dataset =pd.read_csv(r'C:/Users/Ahmed emad/Desktop/AI Assignment/cleaned_car_data.csv')

Car_Dataset.drop(columns=['Model', 'Year', 'Make'], inplace=True)
print("Merged DataFrame shape:", Car_Dataset.shape)
print(Car_Dataset.columns.tolist())
Car_Dataset.to_csv('C:/Users/Ahmed emad/Desktop/AI Assignment/cleaned_car2.csv', index=False)


# print("Merged dataset saved successfully.")