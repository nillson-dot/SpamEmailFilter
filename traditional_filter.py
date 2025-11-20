from filter_func import keyword_filter
from model_evaluator import evaluate_model
import pickle

with open('datas/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('datas/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

def main():
    # 傳統 Filter 預測與評估
    print("\n--- 傳統 Filter 性能評估 ---")
    y_pred_traditional = keyword_filter(X_test)
    evaluate_model(y_test, y_pred_traditional)

if __name__ == "__main__":
    main()