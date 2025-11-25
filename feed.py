from filter_func import keyword_filter
import pickle

with open('datas/lr_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('datas/tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

mailpath = "datas/data.txt"
with open(mailpath, 'r', encoding='utf-8') as file:
    new_raw_mail = file.read()

new_emails = [new_raw_mail]
print(f"郵件內容: {new_raw_mail}")

print("-------傳統 Filter-------")

traditional_prediction = keyword_filter(new_emails)

if traditional_prediction == True:
    status = "垃圾郵件 (SPAM)"
else:
    status = "合法文件 (HAM)"

print(f"預測結果: {status}\n")



new_email_features = loaded_vectorizer.transform(new_emails)

prediction = loaded_model.predict(new_email_features)
prediction_proba = loaded_model.predict_proba(new_email_features)

if prediction[0] == True:
    status = "垃圾郵件 (SPAM)"
else:
    status = "合法文件 (HAM)"

print("--- Fitted Model ---")
print(f"預測結果: {status}")
print(f"預測機率 (Ham/Spam): {prediction_proba[0]}")
