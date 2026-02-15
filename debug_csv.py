
import pandas as pd

def debug_csv_columns():
    csv_path = "d:/Projects/IdealQuant/data/VIP_X030T_1dk_.csv"
    try:
        # Try CP1254 first
        print("--- CP1254 ---")
        df = pd.read_csv(csv_path, sep=';', decimal=',', encoding='cp1254', nrows=5)
        print("Columns:", df.columns.tolist())
        print(df.head())
        
        # Try UTF-8-SIG (BOM check)
        print("\n--- UTF-8-SIG ---")
        df2 = pd.read_csv(csv_path, sep=';', decimal=',', encoding='utf-8-sig', nrows=5)
        print("Columns:", df2.columns.tolist())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_csv_columns()
