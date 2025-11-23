# 使用官方 Python 基礎鏡像
FROM python:3.11-slim

# 將工作目錄設定為 /app
WORKDIR /app

# 將 requirements.txt 複製到工作目錄
COPY requirements.txt .

# 安裝所有 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 將應用程式碼和靜態檔案複製到工作目錄 (確保 static/GGEE.jpg 被包含)
COPY . .

# 這是關鍵：Streamlit 需要一個可訪問的靜態檔案路徑。
# 我們將使用 --server.folder-path 來指向靜態檔案。
# Render 服務器通常運行在 Port $PORT，但在 Dockerfile 中通常指定為 8501
EXPOSE 8501

# 啟動 Streamlit 應用程式
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.enableCORS=true", \
            "--server.enableXsrfProtection=false"]
