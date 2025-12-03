from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def print_specific_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # --- 1. 合法郵件（Ham）的精準度 ---
    ham_precision = report['0']['precision']

    # --- 2. 垃圾郵件（Spam）的召回率 ---
    spam_recall = report['1']['recall']

    # --- 3. 提取 F1-Score ---
    ham_f1 = report['0']['f1-score']
    spam_f1 = report['1']['f1-score']

    print(f"合法郵件Precision: {ham_precision:.4f} (愈高愈好)")
    print(f"垃圾郵件Recall: {spam_recall:.4f} (愈高愈好)")
    print(f"合法郵件 F1-Score: {ham_f1:.4f} (愈高愈好)")
    print(f"垃圾郵件 F1-Score: {spam_f1:.4f} (愈高愈好)")

def evaluate_model(y_test, y_pred):
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # precision, recall, F1-Score
    print("\n(Classification Report):")
    print(classification_report(y_test, y_pred))
    print_specific_metrics(y_test, y_pred)

    # Confusion Matrix
    #  [合法郵件且預測正確 (TN), 垃圾郵件但預測錯誤 (FN)]
    #  [垃圾郵件且預測正確 (TP), 合法郵件但預測錯誤 (FP)]
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n")
    print(f"合法郵件：[預測正確 (TN): {cm[0][0]}, 預測錯誤 (FN): {cm[1][0]}]")
    print(f"垃圾郵件：[預測正確 (TP): {cm[1][1]}, 預測錯誤 (FP): {cm[0][1]}]\n")