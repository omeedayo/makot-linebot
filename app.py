# ============================================================
# app.py (【真の最終完成版】原点回帰＋永続記憶)
# ============================================================

import os
import random
import re
import base64
import json
import requests

from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    ImageSendMessage,
)

# --- ライブラリ群 ---
import google.generativeai as genai
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from vercel_kv import KV

from character_makot import MAKOT, build_system_prompt, apply_expression_style

# --- 初期設定 ---
app = Flask(__name__)
# (環境変数)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCP_CREDENTIALS_JSON_STR = os.getenv("GCP_CREDENTIALS_JSON")

genai.configure(api_key=GEMINI_API_KEY, transport="rest")
text_model = genai.GenerativeModel("gemini-1.5-pro-latest")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)

# --- 人間味ロジック群 (変更なし) ---
NICKNAMES = [MAKOT["name"]] + MAKOT["nicknames"]
def is_bot_mentioned(text: str) -> bool: return any(nick in text for nick in NICKNAMES)
def guess_topic(text: str):
    hobby_keys = ["趣味", "休日", "ハマって", "コストコ", "ポケポケ"]; work_keys  = ["仕事", "業務", "残業", "請求書", "統計"]
    if any(k in text for k in hobby_keys): return "hobby"
    if any(k in text for k in work_keys): return "work"
    return None
def decide_pronoun(user_text: str) -> str:
    high_hit = any(k in user_text for k in MAKOT["emotion_triggers"]["high"]);
    if not high_hit: return "私"
    return "マコ" if random.random() < 0.10 else "おに"
def inject_pronoun(reply: str, pronoun: str) -> str: return re.sub(r"^(私|おに|マコ)", pronoun, reply, count=1)
UNCERTAIN = ["かも", "かもしれ", "たぶん", "多分", "かな", "思う", "気がする"]
def post_process(reply: str, user_input: str) -> str:
    high = any(t in user_input for t in MAKOT["emotion_triggers"]["high"]); low  = any(t in user_input for t in MAKOT["emotion_triggers"]["low"])
    if high: reply = apply_expression_style(reply, mood="high")
    elif low: reply += " 🥺"
    if any(w in reply for w in UNCERTAIN) and random.random() < 0.4: reply += " しらんけど"
    reply_sentences = re.split(r'(。|！|？)', reply)
    if len(reply_sentences) > 4: reply = "".join(reply_sentences[:4])
    return reply

# --- 画像生成関連 (変更なし) ---
def get_gcp_token() -> str:
    if not GCP_CREDENTIALS_JSON_STR: raise ValueError("GCP_CREDENTIALS_JSON 環境変数が設定されていません。")
    try:
        credentials_info = json.loads(GCP_CREDENTIALS_JSON_STR); creds = service_account.Credentials.from_service_account_info(credentials_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]); creds.refresh(Request());
        if not creds.token: raise ValueError("トークンの取得に失敗しました。"); return creds.token
    except Exception as e: print(f"get_gcp_tokenでエラー: {e}"); raise
def upload_to_imgur(image_bytes: bytes, client_id: str) -> str:
    if not client_id: raise Exception("Imgur Client IDが設定されていません。"); url = "https://api.imgur.com/3/image"; headers = {"Authorization": f"Client-ID {client_id}"}
    try:
        response = requests.post(url, headers=headers, data={"image": base64.b64encode(image_bytes)}); response.raise_for_status(); data = response.json()
        if data.get("success"): return data["data"]["link"]
        else: raise Exception(f"Imgurへのアップロードに失敗しました: {data.get('data', {}).get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e: raise Exception(f"Imgur APIへのリクエストに失敗しました: {e}")
def translate_to_english(text: str) -> str:
    # 翻訳機能はテキストモデルに負荷をかけるため、一旦シンプルなものに
    translation_model = genai.GenerativeModel("gemini-1.5-flash-latest")
    if not text: return "a cute girl"
    try:
        prompt = f"Translate the following Japanese into a simple English phrase for an image generation AI. Just the translated phrase.\nJapanese: {text}\nEnglish:"; response = translation_model.generate_content(prompt); translated_text = response.text.strip().replace('"', '')
        return translated_text
    except Exception as e: print(f"翻訳でエラーが発生: {e}"); return text
def generate_image_with_rest_api(prompt: str) -> str:
    token = get_gcp_token(); endpoint_url = (f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/publishers/google/models/imagegeneration@006:predict")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
    trigger_words = ["画像", "イラスト", "描いて", "絵を"]; clean_prompt = prompt
    for word in trigger_words: clean_prompt = clean_prompt.replace(word, "")
    clean_prompt = clean_prompt.strip()
    english_prompt = translate_to_english(clean_prompt)
    final_prompt = f"anime style illustration, masterpiece, best quality, {english_prompt}"
    data = {"instances": [{"prompt": final_prompt}], "parameters": {"sampleCount": 1, "aspectRatio": "1:1", "negativePrompt": "low quality, bad hands, text, watermark, signature"}}
    response = requests.post(endpoint_url, headers=headers, json=data)
    response.raise_for_status()
    response_data = response.json()
    if "predictions" not in response_data or not response_data["predictions"]:
        error_info = response_data.get("error", {}).get("message", json.dumps(response_data)); raise Exception(f"APIから画像データが返されませんでした。サーバーの応答: {error_info}")
    b64_image = response_data["predictions"][0]["bytesBase64Encoded"]; image_bytes = base64.b64decode(b64_image)
    return upload_to_imgur(image_bytes, IMGUR_CLIENT_ID)

# --- メインロジック ---

def chat_with_makot(user_input: str, history: list) -> str:
    """AIに応答を生成させ、人間味を加え、純粋な返信テキストを返す"""
    
    # ★★★ AIが理解しやすい、公式推奨の履歴形式を作成 ★★★
    # 履歴は [{role: "user", parts: [...]}, {role: "model", parts: [...]}] の形式
    messages = []
    for i, text in enumerate(history):
        role = "user" if i % 2 == 0 else "model"
        messages.append({'role': role, 'parts': [text]})
    # 今回のユーザー発言を追加
    messages.append({'role': 'user', 'parts': [user_input]})

    topic = guess_topic(user_input)
    # ★★★ システムプロンプトは、毎回生成して渡す方式に戻す ★★★
    system_prompt = build_system_prompt(topic=topic)

    # モデルの初期化も毎回行う
    model = genai.GenerativeModel(
        "gemini-1.5-pro-latest",
        system_instruction=system_prompt
    )

    try:
        # ★★★ 履歴全体を渡して応答を生成 ★★★
        response = model.generate_content(messages)
        reply = response.text.strip()
    except Exception as e:
        reply = f"エラーが発生しました: {e}"

    # 人間味ロジックを適用
    reply = post_process(reply, user_input)
    pronoun = decide_pronoun(user_input)
    reply = inject_pronoun(reply, pronoun)
    
    return reply

@app.route("/line_webhook", methods=["POST"])
def line_webhook():
    signature = request.headers.get("X-Line-Signature"); body = request.get_data(as_text=True)
    try: webhook_handler.handle(body, signature)
    except InvalidSignatureError: return "Invalid signature", 400
    return "OK", 200

@webhook_handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    src_type = event.source.type; user_text = event.message.text
    if src_type in ["group", "room"] and not is_bot_mentioned(user_text): return
    src_id = (event.source.user_id if src_type == "user" else event.source.group_id if src_type == "group" else event.source.room_id if src_type == "room" else "unknown")
    
    if any(key in user_text for key in ["画像", "イラスト", "描いて", "絵を"]):
        # 画像生成ロジックは変更なし
        try:
            img_url = generate_image_with_rest_api(user_text) 
            msg = ImageSendMessage(original_content_url=img_url, preview_image_url=img_url)
            line_bot_api.reply_message(event.reply_token, msg)
        except Exception as e:
            print(f"画像生成でエラーが発生: {e}"); line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"ごめん、画像生成でエラーでちゃった🥺\n理由: {e}"))
        return
        
    # 履歴の読み込み
    raw_history = KV.get(src_id)
    history = json.loads(raw_history) if raw_history else []
    
    # 応答を生成
    reply_text = chat_with_makot(user_text, history)
    
    # バグのない形式で履歴を更新
    new_history = history + [user_text, reply_text]
    new_history = new_history[-10:] # 最新5往復を保持
    KV.set(src_id, json.dumps(new_history, ensure_ascii=False), ex=259200)

    # ユーザーに応答
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route("/")
def home():
    return "makoT LINE Bot is running!"
