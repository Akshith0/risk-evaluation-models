import pandas as pd
import numpy as np
import os

INPUT_PATH = "../../data/processed/"
OUTPUT_PATH = "../../data/processed/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def calculate_log_returns(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)
            # Ensure the 'Close' column exists
            if 'close' in df.columns:
                df["close"] = pd.to_numeric(df["close"], errors='coerce')
                # Calculate log returns
                df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
                
                # Save the processed DataFrame to the output folder
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_processed{ext}"
                output_path = os.path.join(OUTPUT_PATH, new_filename)
                df.to_csv(output_path, index=False)
                print(f"Processed {new_filename} and saved to {output_path}")
            else:
                print(f"'close' column not found in {new_filename}. Skipping.")

if __name__ == "__main__":
    calculate_log_returns(INPUT_PATH)