import os
import pandas as pd

def main():
    df = pd.DataFrame()
    df.to_csv('datos.csv', index=False)


if __name__ == "__main__":
    main()