import pandas as pd

csv_file_name = "two-broken-fan-blade-3.csv"

df = pd.read_csv(f'data/CSV/two-broken-fan-blade/{csv_file_name}').dropna().to_numpy()

print(df)

new_df = pd.DataFrame({f"data_{i+1}":frame for i, frame in enumerate(df)})

new_df.to_csv(f'data/radar-motor/two-broken-fan-blade/{csv_file_name}', index=False)
