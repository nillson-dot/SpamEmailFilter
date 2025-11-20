import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def main():
    # 載入資料
    datasets = 'datas/emails.csv'
    df = pd.read_csv(datasets, encoding='latin-1') 
    X = df['text']
    y = df['spam']

    # 切分資料
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 80% 訓練集，20% 測試集

    with open('datas/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('datas/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    with open('datas/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('datas/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)

    print(f"\n訓練集大小: {len(X_train)}")
    print(f"測試集大小: {len(X_test)}")

if __name__ == "__main__":
    main()