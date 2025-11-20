from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import pickle


def create_features(X_train, X_test):
    # 使用TF-IDF 向量化器
    tfidf_vectorizer = TfidfVectorizer(
        stop_words=stopwords.words('english'),
        # 移除 stopwords
        analyzer='word',
        # 移除 symbols
        max_df=0.8
        # 忽略 80% emails 都會出現的詞
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    with open('datas/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    return X_train_tfidf, X_test_tfidf
    # 用訓練集的數據 Fit 向量化器，並用 Fitted 的向量化器 Transform 測試集

def train_model(X_train_tfidf, y_train):
    print("\n--- 開始訓練 Model ---")
    #Initialize Logistic Regression model
    lr_model = LogisticRegression(solver='liblinear', random_state=42)
    lr_model.fit(X_train_tfidf, y_train)

    with open('datas/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
        
    return lr_model


