# ============================================================
# app.py (最終統合版：人間味＋画像生成＋会話履歴＋マルチモーダル対応)
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
    ImageMessage,      # ★ 追加
    StickerMessage,    # ★ 追加
)

# --- AI & Cloud Libraries ---
import google.generativeai as genai
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import redis

# ★★★ あなたの最新版 character_makot をインポート ★★★
from character_makot import MAKOT, build_system_prompt, apply_expression_style

# ------------------------------------------------------------
# Flask & LINE Bot setup
# ------------------------------------------------------------
app = Flask(__name__)

# --- 環境変数 ---
GEMINI_API_KEY            = os.getenv("GEMINI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET")
IMGUR_CLIENT_ID           = os.getenv("IMGUR_CLIENT_ID")
GCP_PROJECT_ID            = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION              = os.getenv("GCP_LOCATION", "us-central1")
GCP_CREDENTIALS_JSON_STR  = os.getenv("GCP_CREDENTIALS_JSON")
REDIS_URL                 = os.getenv("REDIS_URL")

# --- Gemini client (text) ---
# ★ モデル名はご指定の通り変更していません
genai.configure(api_key=GEMINI_API_KEY, transport="rest")
text_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

# --- LINE SDK & Redis client ---
line_bot_api    = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)
if not REDIS_URL:
    raise ValueError("REDIS_URL 環境変数が設定されていません。")
redis_client = redis.from_url(REDIS_URL)

# ------------------------------------------------------------
# 「人間味」ロジック群 & 画像生成関連 (変更なし)
# ------------------------------------------------------------
# (このセクションの関数群は一切変更不要です)
NICKNAMES = [MAKOT["name"]] + MAKOT["nicknames"]
def is_bot_mentioned(text: str) -> bool: return any(nick in text for nick in NICKNAMES)
def guess_topic(text: str):
    hobby_keys = ["趣味", "休日", "ハマって", "コストコ", "ポケポケ"]; work_keys  = ["仕事", "業務", "残業", "請求書", "統計"]
    if any(k in text for k in hobby_keys): return "hobby"
    if any(k in text for k in work_keys): return "work"
    return None
def decide_pronoun(user_text: str) -> str:
    high_hit = any(k in user_text for k in MAKOT["emotion_triggers"]["high"])
    if not high_hit: return "私"
    return "マコ" if random.random() < 0.10 else "おに"
def inject_pronoun(reply: str, pronoun: str) -> str: return re.sub(r"^(私|おに|マコ)", pronoun, reply, count=1)
UNCERTAIN = ["かも", "かもしれ", "たぶん", "多分", "かな", "と思う", "気がする"]
def post_process(reply: str, user_input: str) -> str:
    high = any(t in user_input for t in MAKOT["emotion_triggers"]["high"]); low  = any(t in user_input for t in MAKOT["emotion_triggers"]["low"])
    if high: reply = apply_expression_style(reply, mood="high")
    elif low: reply += " 🥺"
    if any(w in reply for w in UNCERTAIN) and random.random() < 0.4: reply += " しらんけど"
    reply_sentences = re.split(r'(。|！|？)', reply)
    if len(reply_sentences) > 4: reply = "".join(reply_sentences[:4])
    return reply
def get_gcp_token() -> str:
    if not GCP_CREDENTIALS_JSON_STR: raise ValueError("GCP_CREDENTIALS_JSON 環境変数が設定されていません。")
    try:
        credentials_info = json.loads(GCP_CREDENTIALS_JSON_STR); creds = service_account.Credentials.from_service_account_info(credentials_info, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(Request());
        if not creds.token: raise ValueError("トークンの取得に失敗しました。")
        return creds.token
    except Exception as e: print(f"get_gcp_tokenでエラー: {e}"); raise
def upload_to_imgur(image_bytes: bytes, client_id: str) -> str:
    if not client_id: raise Exception("Imgur Client IDが設定されていません。")
    url = "https://api.imgur.com/3/image"; headers = {"Authorization": f"Client-ID {client_id}"}
    try:
        response = requests.post(url, headers=headers, data={"image": base64.b64encode(image_bytes)}); response.raise_for_status(); data = response.json()
        if data.get("success"): return data["data"]["link"]
        else: raise Exception(f"Imgurへのアップロードに失敗しました: {data.get('data', {}).get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e: raise Exception(f"Imgur APIへのリクエストに失敗しました: {e}")
def translate_to_english(text: str) -> str:
    if not text: return "a cute girl"
    try:
        prompt = f"Translate the following Japanese into a simple English phrase for an image generation AI. For example, '猫' -> 'a cat', '空を飛ぶ犬' -> 'a dog flying in the sky'. Do not add any extra explanation. Just the translated phrase.\nJapanese: {text}\nEnglish:"
        response = text_model.generate_content(prompt); translated_text = response.text.strip().replace('"', '')
        return translated_text
    except Exception as e: print(f"翻訳でエラーが発生: {e}"); return text
def generate_image_with_rest_api(prompt: str) -> str:
    token = get_gcp_token(); endpoint_url = (f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/publishers/google/models/imagegeneration@006:predict")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
    trigger_words = ["画像", "イラスト", "描いて", "絵を"]; clean_prompt = prompt
    for word in trigger_words: clean_prompt = clean_prompt.replace(word, "")
    clean_prompt = clean_prompt.strip(); english_prompt = translate_to_english(clean_prompt); final_prompt = f"anime style illustration, masterpiece, best quality, {english_prompt}"
    data = {"instances": [{"prompt": final_prompt}], "parameters": {"sampleCount": 1, "aspectRatio": "1:1", "negativePrompt": "low quality, bad hands, text, watermark, signature"}}
    response = requests.post(endpoint_url, headers=headers, json=data); response.raise_for_status(); response_data = response.json()
    if "predictions" not in response_data or not response_data["predictions"]:
        error_info = response_data.get("error", {}).get("message", json.dumps(response_data)); raise Exception(f"APIから画像データが返されませんでした。サーバーの応答: {error_info}")
    b64_image = response_data["predictions"][0]["bytesBase64Encoded"]; image_bytes = base64.b64decode(b64_image)
    return upload_to_imgur(image_bytes, IMGUR_CLIENT_ID)

# ------------------------------------------------------------
# Main chat logic (変更なし)
# ------------------------------------------------------------
def chat_with_makot(user_input: str, user_id: str) -> str:
    history_key = f"chat_history:{user_id}"; history_json = redis_client.get(history_key)
    history: list[str] = json.loads(history_json) if history_json else []
    history.append(f"ユーザー: {user_input}"); context = "\n".join(history[-12:])
    topic = guess_topic(user_input); system_prompt = build_system_prompt(context, topic=topic)
    try:
        response = text_model.generate_content(system_prompt)
        reply = response.text.strip()
    except Exception as e:
        reply = f"エラーが発生しました: {e}"
    reply = post_process(reply, user_input); pronoun = decide_pronoun(user_input); reply = inject_pronoun(reply, pronoun)
    history.append(f"アシスタント: {reply}"); redis_client.set(history_key, json.dumps(history[-50:]))
    return reply

# ------------------------------------------------------------
# Flask endpoints (★ここから構成を変更)
# ------------------------------------------------------------
@app.route("/line_webhook", methods=["POST"])
def line_webhook():
    signature = request.headers.get("X-Line-Signature"); body = request.get_data(as_text=True)
    try: webhook_handler.handle(body, signature)
    except InvalidSignatureError: return "Invalid signature", 400
    return "OK", 200

# ★ 1. テキストメッセージ専用ハンドラ
@webhook_handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    src_type = event.source.type; user_text = event.message.text
    if src_type in ["group", "room"] and not is_bot_mentioned(user_text): return
    src_id = (event.source.user_id if src_type == "user" else event.source.group_id if src_type == "group" else event.source.room_id if src_type == "room" else "unknown")

    # ★ 画像生成の応答を改善
    if any(key in user_text for key in ["画像", "イラスト", "描いて", "絵を"]):
        try:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="おっけーです！ちょっと待っててくださいね…🥰"))
            img_url = generate_image_with_rest_api(user_text)
            msg = ImageSendMessage(original_content_url=img_url, preview_image_url=img_url)
            line_bot_api.push_message(src_id, msg)
        except Exception as e:
            print(f"画像生成でエラーが発生: {e}")
            line_bot_api.push_message(src_id, TextSendMessage(text=f"ごめんなさい、画像生成の調子が悪い・・・のはおめぇのせいだよ\n理由: {e}"))
        return

    reply_text = chat_with_makot(user_text, user_id=src_id)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

# ★ 2.【新規】画像メッセージ専用ハンドラ
@webhook_handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    try:
        message_content = line_bot_api.get_message_content(event.message.id)
        image_bytes = message_content.content

        makot_prompt = "あなたは後輩女子の『まこT』です。ユーザーから送られてきたこの画像を見て、最高のリアクションを1～2文で返してください！食べ物なら「おいしそう！」、動物なら「かわいい！」など、見たままの感情をテンション高めに表現してください。"
        
        # 既存のtext_modelで画像認識を試行
        response = text_model.generate_content([makot_prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
        reply_text = response.text.strip()
        
        reply_text = post_process(reply_text, "テンション上がる") # まこTらしい表現に加工
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    except Exception as e:
        print(f"画像認識でエラーが発生: {e}")
        # モデルが画像非対応の場合などを考慮したエラーメッセージ
        if "support image" in str(e).lower():
             reply_text = "ごめんなさい、今ちょっと目が悪くて画像が見れないみたいです…🥺 また今度見せてください！"
        else:
             reply_text = "ごめんなさい、画像がうまく見れなかったです…🥺"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

# ★ 3.【新規】スタンプメッセージ専用ハンドラ
@webhook_handler.add(MessageEvent, message=StickerMessage)
def handle_sticker_message(event):
    sticker_map = {
        "11537": {"52002734": "ありがとうございます！うれしいです🥰", "52002748": "おつかれさまです！🙇‍♀️"},
        "11538": {"51626494": "ひえっ…！なにかありましたか！？🥺", "51626501": "ふぁーーーーーーーーーーーｗｗｗｗｗｗｗ"}
    }
    package_id = event.message.package_id; sticker_id = event.message.sticker_id
    reply_text = sticker_map.get(package_id, {}).get(sticker_id)

    if not reply_text:
        reply_text = random.choice(["スタンプありがとうございます！🥰", "そのスタンプかわいいですね！", "お、いいスタンプ！私もほしいです！"])
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route("/")
def home():
    return "まこT LINE Bot is running!"
