# --- app.py (最終完美版) ---

import gradio as gr
import pickle
import numpy as np
import sys

# 嘗試匯入傳統過濾器
try:
    from filter_func import keyword_filter 
except ImportError:
    print("錯誤：找不到 filter_func.py 或 keyword_filter 函數。")
    sys.exit(1)

# ===============================================
# 步驟 1: 載入模型和向量化器
# ===============================================
try:
    with open('datas/lr_model.pkl', 'rb') as f:
        MODEL = pickle.load(f)
    with open('datas/tfidf_vectorizer.pkl', 'rb') as f:
        VECTORIZER = pickle.load(f)
    
    HAM_LABEL = False
    SPAM_LABEL = True

    print("AI 模型與向量化器載入成功。")
except FileNotFoundError:
    print("Eooro：找不到 .pkl 檔案，請確認檔案位置。")
    MODEL = None
    VECTORIZER = None
    
# ===============================================
# 步驟 2: 定義預測邏輯 (強化視覺效果)
# ===============================================

def dual_predict(email_text):

    if not email_text:
        return "請輸入內容...", "N/A", "請輸入內容..."

    # --- A. AI 模型預測 ---
    if MODEL is None or VECTORIZER is None:
        ai_status_md = "## ⚠️ 系統錯誤：模型未載入"
        ai_confidence = "N/A"
    else:
        # 1. 特徵轉換
        new_email_features = VECTORIZER.transform([email_text])

        # 2. 模型預測
        prediction = MODEL.predict(new_email_features)[0]
        prediction_proba = MODEL.predict_proba(new_email_features)[0] 

        # 3. 格式化 AI 結果 (HTML 樣式：大字體 + 置中)
        if prediction == SPAM_LABEL:
            # 垃圾郵件 (紅色系)
            status_text = "垃圾郵件 (SPAM)"
            confidence_value = prediction_proba[1] * 100
            ai_status_md = f"""
            <div style="background-color: #ffe6e6; padding: 20px; border-radius: 12px; border: 2px solid #ff4d4d; text-align: center; margin-bottom: 10px;">
                <h1 style="color: #cc0000; margin: 0; font-size: 32px;">⚠️ {status_text}</h1>
                <p style="color: #cc0000; margin: 5px 0 0 0; font-size: 16px;">(AI 模型判定)</p>
            </div>
            """
            ai_confidence = f"機率: {confidence_value:.2f}%"
        else:
            # 合法郵件 (綠色系)
            status_text = "合法郵件 (HAM)"
            confidence_value = prediction_proba[0] * 100
            ai_status_md = f"""
            <div style="background-color: #d4edda; padding: 20px; border-radius: 12px; border: 3px solid #28a745; text-align: center; margin-bottom: 10px;">
                <h1 style="color: #155724; margin: 0; font-size: 32px; font-weight: bold;">✅ {status_text}</h1>
                <p style="color: #155724; margin: 5px 0 0 0; font-size: 16px; font-weight: bold;">(AI 模型判定 - 安全)</p>
            </div>
            """
            ai_confidence = f"機率: {confidence_value:.2f}%"
        
        

    # --- B. 傳統過濾器預測 ---
    traditional_result = keyword_filter(email_text)

    # 格式化傳統過濾器結果 (HTML 樣式：大字體 + 置中)
    if traditional_result == SPAM_LABEL:
        traditional_output_md = f"""
        <div style="padding: 15px; border: 2px dashed #999; border-radius: 10px; text-align: center;">
            <h2 style="color: #555; margin: 0; font-size: 24px;">❌ 垃圾郵件 (SPAM)</h2>
        </div>
        """
    else:
        traditional_output_md = f"""
        <div style="padding: 15px; border: 2px dashed #28a745; background-color: #f0fff4; border-radius: 10px; text-align: center;">
            <h2 style="color: #155724; margin: 0; font-size: 24px;">✔️ 合法郵件 (HAM)</h2>
        </div>
        """

    return ai_status_md, ai_confidence, traditional_output_md


# ===============================================
# 步驟 3: 建構 UI 介面
# ===============================================

with gr.Blocks(title="AI 郵件防護演示") as demo:
    
    # 標題區
    gr.Markdown(
        """
        # 🛡️ AI 驅動的電子郵件安全防護系統
        ### 比較 **人工智慧模型 (Logistic Regression)** 與 **傳統過濾器** 的偵測差異
        """
    )
    
    with gr.Row():
        # 左側：輸入區
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=12, 
                label="📧 郵件內容輸入", 
                placeholder="請將原始郵件內容貼在這裡進行分析..."
            )
            submit_btn = gr.Button("🔍 開始偵測", variant="primary")
            
            # 範例
            gr.Examples(
                examples=[
                    ["Congratulations! You have won a free iPhone. Click here to claim your prize now!"],
                    ["Hi team, please find the attached meeting minutes for review. Thanks."],
                ],
                inputs=input_text,
                label="快速測試範例"
            )

        # 右側：結果區
        with gr.Column(scale=1):
            gr.Markdown("### 📊 偵測結果分析")
            
            # 1. AI 模型區塊
            with gr.Group():
                gr.Markdown("#### 🤖 AI Model")
                output_ai_status = gr.Markdown() # 這裡會顯示大字體的 HTML
                output_ai_confidence = gr.Textbox(label="信心指標", show_label=False)
            
            # 2. 傳統過濾器區塊
            with gr.Group():
                gr.Markdown("#### 📜 傳統過濾器")
                output_traditional = gr.Markdown() # 這裡會顯示大字體的 HTML
    
    # 綁定按鈕
    submit_btn.click(
        fn=dual_predict,
        inputs=input_text,
        outputs=[output_ai_status, output_ai_confidence, output_traditional]
    )

# 啟動伺服器
if __name__ == "__main__":
    print("正在啟動伺服器...請在瀏覽器輸入 http://127.0.0.1:7860")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)