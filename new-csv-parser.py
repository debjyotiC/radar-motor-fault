import pandas as pd

csv_file_name = "broken-fan-blade-3.csv"

df = pd.read_csv(f'data/CSV/broken-fan-blade/{csv_file_name}').dropna().to_numpy()

new_df = pd.DataFrame({f"data_{i+1}":frame for i, frame in enumerate(df)})

new_df.to_csv(f'data/radar-motor/broken-fan-blade/{csv_file_name}', index=False)
