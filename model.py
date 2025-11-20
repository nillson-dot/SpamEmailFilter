from model_trainer import create_features, train_model
from model_evaluator import evaluate_model
import pickle

def main():
    with open('datas/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('datas/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    with open('datas/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('datas/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    
    # 提取特徵
    X_train_tfidf, X_test_tfidf = create_features(X_train, X_test)
    
    # 訓練模型
    lr_model = train_model(X_train_tfidf, y_train)
    print("Model 訓練完成。")
    
    # Model 預測與評估
    print("\n--- Model 性能評估 ---")
    y_pred = lr_model.predict(X_test_tfidf)
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()