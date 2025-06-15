import os
import json
from flask import Flask
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import requests
import traceback

app = Flask(__name__)

# Vercelから環境変数を読み込む
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCP_CREDENTIALS_JSON_STR = os.getenv("GCP_CREDENTIALS_JSON")

def get_gcp_token():
    """認証情報を元にアクセストークンを取得する関数"""
    if not GCP_CREDENTIALS_JSON_STR:
        raise ValueError("環境変数 'GCP_CREDENTIALS_JSON' が設定されていません。")
    
    credentials_info = json.loads(GCP_CREDENTIALS_JSON_STR)
    creds = service_account.Credentials.from_service_account_info(
        credentials_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    
    creds.refresh(Request()) # トークンをリフレッシュ
    
    if not creds.token:
        raise ValueError("トークンの取得に失敗しました。")
        
    return creds.token

@app.route("/")
def home():
    """
    ルートURLにアクセスした時に、認証の成否を表示する
    """
    try:
        token = get_gcp_token()
        # 認証が成功したら、トークンの一部を表示
        return (f"<h1>認証テスト成功！</h1>"
                f"<p>Google Cloudからアクセストークンを取得できました。</p>"
                f"<p>Token starts with: {token[:12]}...</p>"
                f"<p><a href='/test_api_call'>次にAPI呼び出しテストへ進む</a></p>"), 200
    except Exception as e:
        # 認証でエラーが出たら、詳細なエラー内容を表示
        error_details = traceback.format_exc()
        return (f"<h1>認証テスト失敗...</h1>"
                f"<h2>エラー内容:</h2><pre>{e}</pre>"
                f"<h2>詳細:</h2><pre>{error_details}</pre>"), 500

@app.route("/test_api_call")
def test_api_call():
    """
    実際にVertex AIのエンドポイントにアクセスできるかテストする
    """
    try:
        token = get_gcp_token()
        
        # Vertex AIのモデル一覧を取得するAPIエンドポイント (画像生成より基本的なAPI)
        endpoint_url = (
            f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}"
            f"/locations/{GCP_LOCATION}/models"
        )
        
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.get(endpoint_url, headers=headers)
        
        if response.status_code == 200:
            return (f"<h1>API呼び出しテスト成功！</h1>"
                    f"<p>ステータスコード: {response.status_code}</p>"
                    f"<p>これで画像生成も動くはずです！元のapp.pyに戻してください。</p>"
                    f"<h3>レスポンス:</h3><pre>{response.text}</pre>"), 200
        else:
            return (f"<h1>API呼び出しテスト失敗...</h1>"
                    f"<p>認証は通りましたが、API呼び出しでエラーが発生しました。</p>"
                    f"<h2>ステータスコード: {response.status_code}</h2>"
                    f"<h2>レスポンス:</h2><pre>{response.text}</pre>"), 500

    except Exception as e:
        error_details = traceback.format_exc()
        return (f"<h1>API呼び出しテスト中に予期せぬエラー</h1>"
                f"<h2>エラー内容:</h2><pre>{e}</pre>"
                f"<h2>詳細:</h2><pre>{error_details}</pre>"), 500
