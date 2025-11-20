from sklearn.metrics import classification_report
from model_evaluator import evaluate_model

TRADITIONAL_KEYWORDS = [
    'free', 'prize', 'gift', 'cash', 'money', 'won', 'million', 'urgent', 
    'immediate', 'now', 'act now', 'click here', 'password', 'account', 
    'bank', 'credit card', 'security alert', 'verification', 'login', 
    'update your info', 'viagra', 'cialis', '!!!', '$$$', 'limited time'
    'free', 'prize', 'gift', 'cash', 'money', 'won', 'million', 'urgent', 
    'immediate', 'now', 'act now', 'click here', 'password', 'account', 
    'bank', 'credit card', 'security alert', 'verification', 'login', 
    'update your info', 'viagra', 'cialis', 
    # 新增短语
    'click here to unsubscribe', 'limited time offer', 'you have been selected',
    'congratulations you', 'dear valued customer'
]

def keyword_filter(X_data):
    y_pred_traditional = []
    for email_text in X_data:
        # 關鍵字偵測
        appended = False
        text_lower = email_text.lower()
        for keyword in TRADITIONAL_KEYWORDS:
            if keyword in text_lower:
                appended = True
                y_pred_traditional.append(1)
                break
        if appended:
            continue

        # 大寫字母佔比超過20%
        import re
        text_clean = re.sub(r'[^a-zA-Z]', '', email_text)
        if not text_clean:
            y_pred_traditional.append(0)
            continue
        num_uppercase = sum(1 for char in email_text if char.isupper())
        total_chars = len(text_clean)
        
        if total_chars > 0 and (num_uppercase / total_chars) > 0.20:
            y_pred_traditional.append(1)
            continue
        
        # 超過 3 個連結
        link_count = text_lower.count('http') + text_lower.count('www.')
        if link_count >= 3:
            y_pred_traditional.append(1)
            continue

        y_pred_traditional.append(0)
    return y_pred_traditional
