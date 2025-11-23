# 使用官方 Python 基礎鏡像 (選擇穩定版本)
FROM python:3.11-slim

# 將工作目錄設定為 /app
WORKDIR /app

# 將 requirements.txt 複製到工作目錄
# 這裡我們使用完整的名稱，並假設它可以被正確安裝在 Docker 環境中
COPY requirements.txt .

# 安裝所有 Python 依賴
# 使用 --no-cache-dir 確保安裝乾淨
RUN pip install --no-cache-dir -r requirements.txt

# 將應用程式碼複製到工作目錄
COPY . .

# 定義啟動 Streamlit 應用程式的指令
# Render 預設使用 $PORT 環境變數，通常是 10000
EXPOSE 8501 

# 啟動 Streamlit 應用程式
# 使用 $PORT 環境變數來確保服務器監聽正確的端口
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=true", "--server.address=0.0.0.0"]
