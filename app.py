# ============================================================
# app.py (ステップ5: 定期メッセージ改善版 + 画像生成ルートB(nano-banana)対応 + /judge 最小改善版)
# ============================================================

import os
import random
import re
import base64
import json
import requests
import time
import textwrap
import uuid
from typing import Optional
import datetime  # ★★★ 祝日判定のために追加 ★★★
import jpholiday  # ★★★ 祝日判定のために追加 ★★★

from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    ImageSendMessage, ImageMessage, StickerMessage
)

# --- AI & Cloud Libraries ---
import google.generativeai as genai
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import redis
import pinecone
from dotenv import load_dotenv

# --- 他のPythonファイルからインポート ---
from character_makot import MAKOT, build_system_prompt, apply_expression_style

# ------------------------------------------------------------
# 初期化処理
# ------------------------------------------------------------
load_dotenv('.env.development.local')
app = Flask(__name__)

# --- 環境変数 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCP_CREDENTIALS_JSON_STR = os.getenv("GCP_CREDENTIALS_JSON")
REDIS_URL = os.getenv("REDIS_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "gemini-3-flash-preview")
VERTEX_EMBEDDING_MODEL = os.getenv("VERTEX_EMBEDDING_MODEL", "text-multilingual-embedding-002")
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", 0.55))
CRON_SECRET = os.getenv("CRON_SECRET")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# ★★★ 画像生成モデル（Gemini API側） ★★★
# Vercel環境変数に IMAGE_MODEL_NAME を入れれば切替できる
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "nano-banana-pro-preview")

# --- 各種クライアントの初期化 ---
genai.configure(api_key=GEMINI_API_KEY, transport="rest")
text_model = genai.GenerativeModel(TEXT_MODEL_NAME)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)

if not REDIS_URL:
    raise ValueError("REDIS_URL 環境変数が設定されていません。")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

gcp_token_cache = {"token": None, "expires_at": 0}

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError("Pineconeの環境変数(API_KEY, INDEX_NAME)が設定されていません。")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# ------------------------------------------------------------
# ベクトル化 & RAG関連関数
# ------------------------------------------------------------
def get_gcp_token() -> str:
    if gcp_token_cache["token"] and time.time() < gcp_token_cache["expires_at"]:
        return gcp_token_cache["token"]
    if not GCP_CREDENTIALS_JSON_STR:
        raise ValueError("GCP_CREDENTIALS_JSON 環境変数が設定されていません。")
    try:
        credentials_info = json.loads(GCP_CREDENTIALS_JSON_STR)
        creds = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(Request())
        if not creds.token:
            raise ValueError("トークンの取得に失敗しました。")
        gcp_token_cache["token"] = creds.token
        gcp_token_cache["expires_at"] = time.time() + 3300
        return creds.token
    except Exception as e:
        print(f"get_gcp_tokenでエラー: {e}")
        raise

def _get_vertex_embedding(text: str, task_type: str) -> list[float]:
    """Vertex AIのEmbeddingモデルを呼び出す共通関数"""
    if not text:
        return []
    try:
        token = get_gcp_token()
        endpoint_url = (
            f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}"
            f"/locations/{GCP_LOCATION}/publishers/google/models/{VERTEX_EMBEDDING_MODEL}:predict"
        )
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
        data = {"instances": [{"content": text, "task_type": task_type}]}
        response = requests.post(endpoint_url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        if "predictions" not in response_data or not response_data["predictions"]:
            print(f"Vertex AIからembeddingが返されませんでした: {response_data}")
            return []
        embedding = response_data["predictions"][0]["embeddings"]["values"]
        return embedding
    except Exception as e:
        print(f"Vertex AI ベクトル化エラー: {e}")
        return []

def get_embedding(text: str) -> list[float]:
    """テキストをベクトルに変換する（通常会話の記憶検索用）"""
    return _get_vertex_embedding(text, task_type="RETRIEVAL_DOCUMENT")

def get_qa_embedding(text: str, task_type="RETRIEVAL_QUERY") -> list[float]:
    """Q&A検索用のテキストをベクトルに変換する"""
    return _get_vertex_embedding(text, task_type=task_type)

def summarize_and_store_memory(user_id: str, history: list[str]):
    """会話を要約し、ベクトル化してPineconeに長期記憶として保存する"""
    recent_talk = "\n".join(history[-4:])
    if len(recent_talk) < 50:
        return

    summary_prompt = textwrap.dedent(f"""
        あなたはユーザーとの会話の要約担当です。以下の会話から、ユーザーの個人的な情報（名前、好み、最近の出来事、ペット、悩み、計画など）を抽出し、簡潔な箇条書きのメモとして1～2行で要約してください。重要な情報が含まれていない場合は、必ず「特になし」とだけ出力してください。
        ---
        会話:
        {recent_talk}
        ---
        要約:
    """)
    try:
        response = text_model.generate_content(summary_prompt, tools=[])
        summary = response.text.strip()

        if summary and "特になし" not in summary:
            vector = get_embedding(summary)
            if not vector:
                return

            memory_id = str(uuid.uuid4())
            metadata = {"user_id": user_id, "text": summary, "created_at": time.time()}
            pinecone_index.upsert(
                vectors=[(memory_id, vector, metadata)],
                namespace="conversation-memory"
            )
            print(f"[{user_id}] の新しい記憶をベクトルDBに保存しました: {summary}")
    except Exception as e:
        print(f"記憶の保存処理でエラー: {e}")

# ------------------------------------------------------------
# Q&Aモードと通常会話モードの処理
# ------------------------------------------------------------
QA_SYSTEM_PROMPT = textwrap.dedent("""
    あなたは、後輩女子『まこT』として、提供された参考情報に【基づいてのみ】ユーザーの質問に回答するアシスタントです。
    あなたの役割は、参考情報の内容を分かりやすく、親しみやすい口調で要約して伝えることです。

    【重要ルール】
    - 必ず参考情報に含まれる事実だけを使って回答してください。
    - 参考情報に答えがない場合や、関連性が低い場合は、絶対に推測で答えてはいけません。代わりに「うーん、その情報は見当たらないですね…！ごめんなさい🥺」と正直に回答してください。
    - 回答の最後に見つかった出典（source）をすべて、 `(参考: ファイル名1, ファイル名2)` のようにカンマ区切りで付け加えてください。

    【参考情報】
    {context}

    【ユーザーの質問】
    {question}

    以上のルールを厳格に守り、『まこT』として回答してください：
""")

# ------------------------------------------------------------
# /judge（最小改善: 監査官モード + 出力固定 + 褒め禁止）
# ------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = textwrap.dedent("""
あなたは「判断監査官」です。ユーザーの結論に同意したり称賛したりしません。
目的は、判断が成立するための暗黙前提・破綻点・事故の形・ガードレールを冷静に列挙することです。

ルール:
- 断定よりも「前提」「不確実性」「検証方法」を優先
- ユーザーが書いた前提を繰り返し過ぎない。足りない前提を追加する
- ふわっと褒めない。感想禁止
- 出力は必ず指定フォーマットに従う（日本語）

出力フォーマット:
【見えてる前提】
- ...

【見えてない前提（追加で列挙）】
- ...

【特に危険な前提（上位3つ）】
1) ...（なぜ危険か / 崩れ方）
2) ...
3) ...

【崩れた時の事故の形（具体例）】
- ...

【ガードレール（仕組みで止める案）】
- ...

【次に確認すべき質問（3つ）】
- ...
""").strip()

judge_model = genai.GenerativeModel(
    TEXT_MODEL_NAME,
    system_instruction=JUDGE_SYSTEM_PROMPT
)

def _handle_judge_chat(user_input: str, user_id: str) -> str:
    """/judge モードの処理を担当する（通常のpost_processを通さない）"""
    print(f"[{user_id}] JUDGEモードで実行します。")
    try:
        prompt = textwrap.dedent(f"""
        ユーザー入力:
        {user_input}

        指示:
        上の内容を監査官として分解し、指定フォーマットで出力してください。
        """).strip()

        response = judge_model.generate_content(prompt, tools=[])
        reply = response.text.strip()

        # 記号除去（見た目調整）
        reply = re.sub(r'[\*`＊∗]+', '', reply)

        # 長すぎ保険
        if len(reply) > 1600:
            reply = reply[:1600] + "…（長いので途中まで）"

        return reply
    except Exception as e:
        print(f"JUDGEモードでエラー: {e}")
        return "監査官モードでエラーが出ました。入力を短くしてもう一回ください。"

# ------------------------------------------------------------
# Q&A / 検索 / 通常会話
# ------------------------------------------------------------
def expand_query(question: str) -> list[str]:
    """LLMを使って質問を複数の表現に拡張する"""
    prompt = textwrap.dedent(f"""
        ユーザーの質問を、ベクトル検索でよりヒットしやすくなるように、異なる視点から3つの類義質問や検索キーワードに書き換えてください。
        元の質問も必ず含めてください。箇条書き（ハイフン区切り）で、説明は不要です。
        質問: {question}
        書き換え:
    """)
    try:
        response = text_model.generate_content(prompt, tools=[])
        queries = [line.strip().lstrip('- ') for line in response.text.strip().split('\n') if line.strip()]
        return list(set(queries))
    except Exception as e:
        print(f"クエリ拡張エラー: {e}")
        return [question]

def _handle_qa_request(user_input: str, user_id: str) -> str:
    """Q&Aモードの処理を担当する"""
    print(f"[{user_id}] Q&Aモードで実行します。")
    try:
        expanded_queries = expand_query(user_input)
        print(f"  [クエリ拡張] 元の質問: '{user_input}' -> 拡張後: {expanded_queries}")

        all_matches = {}
        for query in expanded_queries:
            query_vector = get_qa_embedding(query)
            if not query_vector:
                continue

            query_response = pinecone_index.query(
                vector=query_vector, top_k=3, namespace="company-docs", include_metadata=True
            )
            for match in query_response['matches']:
                if match.id not in all_matches or match.score > all_matches[match.id].score:
                    all_matches[match.id] = match

        sorted_matches = sorted(all_matches.values(), key=lambda x: x.score, reverse=True)
        context_chunks, sources = [], set()

        print("\n--- 統合後の検索結果 ---")
        for match in sorted_matches[:5]:
            print(
                f"  [検索結果] Score: {match.score:.4f}, "
                f"Source: {match.metadata['source']}, "
                f"Chapter: {match.metadata.get('chapter', 'N/A')}"
            )
            if match.score > RAG_SCORE_THRESHOLD:
                context_chunks.append(
                    f"【出典: {match.metadata['source']} / 章: {match.metadata.get('chapter', 'N/A')}】\n"
                    f"{match.metadata['text']}"
                )
                sources.add(match.metadata['source'])

        if not context_chunks:
            return "うーん、その情報は見当たらないですね…！ごめんなさい🥺"

        context_str = "\n---\n".join(context_chunks)
        source_str = f"(参考: {', '.join(sorted(list(sources)))})"
        prompt = QA_SYSTEM_PROMPT.format(context=context_str, question=user_input)
        response = text_model.generate_content(prompt, tools=[])
        reply = response.text.strip()

        if "ごめんなさい" not in reply and "参考:" not in reply:
            reply += f" {source_str}"
        reply = re.sub(r'[\*`＊∗]+', '', reply)
        return reply
    except Exception as e:
        print(f"Q&A処理エラー: {e}")
        return "ごめんなさい、なんだかシステムが不調みたいです…。もう一度試してみてください！"

def _handle_normal_chat(user_input: str, user_id: str) -> str:
    """通常会話モードの処理を担当する"""
    print(f"[{user_id}] 通常会話モードで実行します。")
    history_key = f"chat_history:{user_id}"
    history_json = redis_client.get(history_key)
    history: list[str] = json.loads(history_json) if history_json else []

    long_term_memory = None
    try:
        input_vector = get_embedding(user_input)
        if input_vector:
            query_response = pinecone_index.query(
                vector=input_vector, top_k=3, namespace="conversation-memory",
                filter={"user_id": user_id}, include_metadata=True
            )
            relevant_memories = [m['metadata']['text'] for m in query_response['matches'] if m['score'] > 0.7]
            if relevant_memories:
                long_term_memory = "\n".join(f"- {mem}" for mem in relevant_memories)
                print(f"[{user_id}] の関連記憶を検索: {long_term_memory}")
    except Exception as e:
        print(f"記憶の検索エラー: {e}")

    history.append(f"ユーザー: {user_input}")
    context = "\n".join(history[-12:])
    topic = guess_topic(user_input)
    system_prompt = build_system_prompt(context, topic, user_id, long_term_memory)

    try:
        response = text_model.generate_content(system_prompt, tools=[])
        reply = response.text.strip()
    except Exception as e:
        reply = f"エラーが発生しました: {e}"

    reply = post_process(reply, user_input)
    pronoun = decide_pronoun(user_input)
    reply = inject_pronoun(reply, pronoun)

    history.append(f"アシスタント: {reply}")
    redis_client.set(history_key, json.dumps(history[-50:]))
    summarize_and_store_memory(user_id, history)

    return reply

def _handle_search_chat(user_input: str, user_id: str) -> str:
    """Google Custom Search API を使ってWeb検索を行う"""
    print(f"[{user_id}] 検索モードで実行します。 検索語: '{user_input}'")

    if not GOOGLE_SEARCH_API_KEY or not SEARCH_ENGINE_ID:
        return "ごめんなさい、検索機能が設定されていないみたいです…🥺"

    try:
        print("  [ステップ1] Google Custom Search APIで検索を実行します...")
        search_service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
        result = search_service.cse().list(q=user_input, cx=SEARCH_ENGINE_ID, num=5).execute()

        if 'items' not in result or not result['items']:
            return "うーん、関連する情報が見つかりませんでした…！ごめんなさい🥺"

        context_chunks = []
        for i, item in enumerate(result['items']):
            title = item.get('title', 'No Title')
            snippet = item.get('snippet', 'No Snippet').replace('\n', '')
            link = item.get('link', '')
            context_chunks.append(f"【検索結果{i+1}】\nタイトル: {title}\n要約: {snippet}\nURL: {link}")

        search_context = "\n\n".join(context_chunks)
        print("  [ステップ1完了] 検索コンテキストを生成しました。")

        print("  [ステップ2] 要約を生成します...")
        summarize_prompt = textwrap.dedent(f"""
        あなたは後輩女子『まこT』です。提供された以下の【Web検索結果】を基に、ユーザーの質問に親しみやすく、分かりやすく要約して回答してください。

        【重要な指示】
        - 【Web検索結果】に書かれている情報だけを使ってください。
        - 重要なポイントを2～3点に絞って、箇条書きなどで分かりやすく要約します。
        - 回答は、まこTの明るく親しみやすいキャラクターで、150文字程度で簡潔にまとめてください。
        - 検索結果が乏しい場合は、無理にまとめず見つかった情報を素直に伝える形で回答してください。

        【Web検索結果】
        {search_context}

        【元のユーザーの質問】
        {user_input}
        """)

        response = text_model.generate_content(summarize_prompt, tools=[])
        reply = response.text.strip()

        reply = post_process(reply, "テンション上がる", is_search=True)
        reply = re.sub(r'[\*`＊∗]+', '', reply)
        return reply

    except Exception as e:
        print(f"検索チャットでエラーが発生: {e}")
        return "ごめんなさい、検索中にエラーが起きちゃいました…🥺 もう一度試してみてください！"

def chat_with_makot(user_input: str, user_id: str) -> str:
    """ユーザー入力に応じてJUDGE、検索、Q&A、通常会話のモードを振り分ける"""

    # 0. JUDGEモードの判定（最優先）
    # "/judge" 単体でも "/judge " でも拾う
    if user_input.startswith("/judge") or user_input.startswith("/j"):
        # 先頭コマンドだけ剥がす
        if user_input.startswith("/judge"):
            text = user_input[len("/judge"):].strip()
        else:
            text = user_input[len("/j"):].strip()

        if not text:
            return (
                "/judge のあとに貼るだけでOK。\n"
                "例:\n"
                "/judge\n"
                "目的: AI活用範囲の整理\n"
                "前提: インフラはミス不可、AIは誤る\n"
                "問い: 暗黙前提と事故の形とガードレールは？"
            )
        return _handle_judge_chat(text, user_id)

    # 1. 検索モードの判定
    search_keywords = ["調べて", "しらべて", "/search "]
    for keyword in search_keywords:
        if user_input.startswith(keyword):
            question = user_input.replace(keyword, "", 1).strip()
            if not question:
                break
            return _handle_search_chat(question, user_id)

    # 2. Q&Aモードの判定
    qa_keywords = ["仕事", "QA ", "/qa "]
    for keyword in qa_keywords:
        if user_input.startswith(keyword):
            question = user_input.replace(keyword, "", 1).strip()
            if not question:
                break
            return _handle_qa_request(question, user_id)

    # 3. どちらでもなければ通常会話
    return _handle_normal_chat(user_input, user_id=user_id)

# ------------------------------------------------------------
# ユーティリティ & Webhookハンドラ
# ------------------------------------------------------------
def is_bot_mentioned(text: str) -> bool:
    return any(nick in text for nick in [MAKOT["name"]] + MAKOT["nicknames"])

def guess_topic(text: str):
    hobby_keys = ["趣味", "休日", "ハマって", "コストコ", "ポケポケ"]
    work_keys = ["仕事", "業務", "残業", "請求書", "統計"]
    if any(k in text for k in hobby_keys):
        return "hobby"
    if any(k in text for k in work_keys):
        return "work"
    return None

def decide_pronoun(user_text: str) -> str:
    return "マコ" if random.random() < 0.10 else "おに" if any(k in user_text for k in MAKOT["emotion_triggers"]["high"]) else "私"

def inject_pronoun(reply: str, pronoun: str) -> str:
    return re.sub(r"^(私|おに|マコ)", pronoun, reply, count=1)

UNCERTAIN = ["かも", "かもしれ", "たぶん", "多分", "かな", "と思う", "気がする"]

def post_process(reply: str, user_input: str, is_search: bool = False) -> str:
    if any(t in user_input for t in MAKOT["emotion_triggers"]["high"]):
        reply = apply_expression_style(reply, mood="high")
    elif any(t in user_input for t in MAKOT["emotion_triggers"]["low"]):
        reply += " 🥺"

    reply = re.sub(r'[\*`＊∗]+', '', reply)

    if any(w in reply for w in UNCERTAIN) and random.random() < 0.4:
        reply += " しらんけど"

    if not is_search:
        reply_sentences = re.split(r'([。！？])', reply)
        if len(reply_sentences) > 5:
            processed_reply = "".join(reply_sentences[:4])
            reply = processed_reply

    return reply

def upload_to_imgur(image_bytes: bytes, client_id: str) -> str:
    if not client_id:
        raise Exception("Imgur Client IDが設定されていません。")
    url = "https://api.imgur.com/3/image"
    headers = {"Authorization": f"Client-ID {client_id}"}
    try:
        response = requests.post(url, headers=headers, data={"image": base64.b64encode(image_bytes)})
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data["data"]["link"]
        raise Exception(f"Imgurへのアップロードに失敗しました: {data.get('data', {}).get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Imgur APIへのリクエストに失敗しました: {e}")

def translate_to_english(text: str) -> str:
    if not text:
        return "a cute girl"
    try:
        prompt = (
            "Translate the following Japanese into a simple English phrase for an image generation AI. "
            "Just the translated phrase.\n"
            f"Japanese: {text}\nEnglish:"
        )
        response = text_model.generate_content(prompt, tools=[])
        return response.text.strip().replace('"', '')
    except Exception as e:
        print(f"翻訳でエラーが発生: {e}")
        return text

# ------------------------------------------------------------
# 画像生成（ルートB: Gemini API / nano-banana）
# ------------------------------------------------------------
def generate_image_with_gemini_api(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY が設定されていません。")

    trigger_words = ["画像", "イラスト", "描いて", "絵を"]
    clean_prompt = re.sub("|".join(trigger_words), "", prompt).strip()
    if not clean_prompt:
        clean_prompt = "a cute cat"

    english_prompt = translate_to_english(clean_prompt)
    final_prompt = f"anime style illustration, masterpiece, best quality, {english_prompt}"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{IMAGE_MODEL_NAME}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [{
            "parts": [{"text": final_prompt}]
        }],
        "generationConfig": {
            "responseModalities": ["Image"],
            "imageConfig": {
                "aspectRatio": "1:1",
                "imageSize": "1K"
            }
        }
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()

    try:
        parts = data["candidates"][0]["content"]["parts"]
    except Exception:
        raise Exception(f"画像レスポンスの形式が想定外です: {data}")

    b64 = None
    for p in parts:
        if "inlineData" in p and isinstance(p["inlineData"], dict):
            b64 = p["inlineData"].get("data")
            if b64:
                break

    if not b64:
        raise Exception(f"画像データが返ってきませんでした: {data}")

    image_bytes = base64.b64decode(b64)
    return upload_to_imgur(image_bytes, IMGUR_CLIENT_ID)

# 旧: Vertex画像生成（いまは使わないが残しておく）
def generate_image_with_rest_api(prompt: str) -> str:
    token = get_gcp_token()
    endpoint_url = (
        f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}"
        f"/publishers/google/models/imagegeneration@006:predict"
    )
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}

    trigger_words = ["画像", "イラスト", "描いて", "絵を"]
    clean_prompt = re.sub("|".join(trigger_words), "", prompt).strip()
    english_prompt = translate_to_english(clean_prompt)
    final_prompt = f"anime style illustration, masterpiece, best quality, {english_prompt}"

    data = {
        "instances": [{"prompt": final_prompt}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "1:1",
            "negativePrompt": "low quality, bad hands, text, watermark, signature"
        }
    }
    response = requests.post(endpoint_url, headers=headers, json=data)
    response.raise_for_status()
    response_data = response.json()
    if "predictions" not in response_data or not response_data["predictions"]:
        error_info = response_data.get("error", {}).get("message", json.dumps(response_data))
        raise Exception(f"APIから画像データが返されませんでした。サーバーの応答: {error_info}")

    b64_image = response_data["predictions"][0]["bytesBase64Encoded"]
    image_bytes = base64.b64decode(b64_image)
    return upload_to_imgur(image_bytes, IMGUR_CLIENT_ID)

@app.route("/line_webhook", methods=["POST"])
def line_webhook():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    try:
        webhook_handler.handle(body, signature)
    except InvalidSignatureError:
        return "Invalid signature", 400
    return "OK", 200

@app.route("/push/monday", methods=["GET", "POST"])
def push_monday_message():
    auth_header = request.headers.get('Authorization')
    if not CRON_SECRET or auth_header != f"Bearer {CRON_SECRET}":
        print("Unauthorized cron request")
        return "Unauthorized", 401

    user_ids = redis_client.smembers("users")
    if not user_ids:
        print("送信対象ユーザーがいません。")
        return "No users to send message to.", 200

    today = datetime.date.today()

    monday_normal_messages = [
        "月曜日ですね！今週も頑張っていきましょー！💪",
        "げつようび…！無理せず、ぼちぼちいきましょ～！🥺",
        "新しい一週間のはじまりですね！ファイトです！🔥"
    ]
    monday_holiday_messages = [
        "今日はお休みですね！ゆっくりリフレッシュしてくださいね🥰",
        "祝日ですね！良い一日を～！✨"
    ]

    if jpholiday.is_holiday(today):
        text = random.choice(monday_holiday_messages)
    else:
        text = random.choice(monday_normal_messages)

    message = TextSendMessage(text=text)

    try:
        line_bot_api.multicast(list(user_ids), message)
        print(f"{len(user_ids)} 人のユーザーに月曜日のメッセージを送信しました。")
        return "OK", 200
    except Exception as e:
        print(f"LINEへのマルチキャスト送信でエラーが発生: {e}")
        return "Failed to send message to LINE", 500

@app.route("/push/friday", methods=["GET", "POST"])
def push_friday_message():
    auth_header = request.headers.get('Authorization')
    if not CRON_SECRET or auth_header != f"Bearer {CRON_SECRET}":
        print("Unauthorized cron request")
        return "Unauthorized", 401

    user_ids = redis_client.smembers("users")
    if not user_ids:
        print("送信対象ユーザーがいません。")
        return "No users to send message to.", 200

    today = datetime.date.today()

    is_today_holiday = jpholiday.is_holiday(today)

    monday = today + datetime.timedelta(days=3)
    is_monday_holiday = jpholiday.is_holiday(monday)

    friday_normal_messages = [
        "金曜日！今週もお疲れ様でした🍻 よい週末を〜🥰",
        "華金ですね！おつかれさまです！🎉",
        "やっと週末…！今週もよく頑張りましたね！偉いです！🥺"
    ]
    friday_holiday_messages = [
        "明日から連休ですね！ゆっくり羽を伸ばしてください～！✨",
        "３連休だー！やったー！良い休日を！🥳"
    ]

    if is_today_holiday or is_monday_holiday:
        text = random.choice(friday_holiday_messages)
    else:
        text = random.choice(friday_normal_messages)

    message = TextSendMessage(text=text)

    try:
        line_bot_api.multicast(list(user_ids), message)
        print(f"{len(user_ids)} 人のユーザーに金曜日のメッセージを送信しました。")
        return "OK", 200
    except Exception as e:
        print(f"LINEへのマルチキャスト送信でエラーが発生: {e}")
        return "Failed to send message to LINE", 500

@webhook_handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    redis_client.sadd("users", user_id)
    user_text = event.message.text

    if event.source.type in ["group", "room"] and not is_bot_mentioned(user_text):
        return

    # 画像生成
    if any(key in user_text for key in ["画像", "イラスト", "描いて", "絵を"]):
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="おっけーです！ちょっと待っててくださいね…🥰")
            )
            img_url = generate_image_with_gemini_api(user_text)
            msg = ImageSendMessage(original_content_url=img_url, preview_image_url=img_url)
            line_bot_api.push_message(user_id, msg)
        except Exception as e:
            print(f"画像生成でエラーが発生: {e}")
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text=f"ごめんなさい、画像生成の調子が悪いみたいです…\n理由: {e}")
            )
        return

    # 通常テキスト
    reply_text = chat_with_makot(user_text, user_id=user_id)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@webhook_handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    user_id = event.source.user_id
    redis_client.sadd("users", user_id)
    try:
        message_content = line_bot_api.get_message_content(event.message.id)
        image_bytes = message_content.content
        makot_prompt = "あなたは後輩女子の『まこT』です。ユーザーから送られてきたこの画像を見て、最高のリアクションを1～2文で返してください！食べ物なら「おいしそう！」、動物なら「かわいい！」など、見たままの感情をテンション高めに表現してください。"
        response = text_model.generate_content([makot_prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
        reply_text = response.text.strip()
        reply_text = post_process(reply_text, "テンション上がる")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
    except Exception as e:
        print(f"画像認識でエラーが発生: {e}")
        if "support image" in str(e).lower() or "image format" in str(e).lower():
            reply_text = "ごめんなさい、今ちょっと目が悪くて画像が見れないみたいです…🥺 また今度見せてください！"
        else:
            reply_text = "ごめんなさい、画像がうまく見れなかったです…🥺"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@webhook_handler.add(MessageEvent, message=StickerMessage)
def handle_sticker_message(event):
    user_id = event.source.user_id
    redis_client.sadd("users", user_id)
    sticker_map = {
        "11537": {
            "52002734": "ありがとうございます！うれしいです🥰",
            "52002748": "おつかれさまです！🙇‍♀️"
        },
        "11538": {
            "51626494": "ひえっ…！なにかありましたか！？🥺",
            "51626501": "ふぁーーーーーーーーーーーｗｗｗｗｗｗｗ"
        }
    }
    package_id = event.message.package_id
    sticker_id = event.message.sticker_id
    reply_text = sticker_map.get(str(package_id), {}).get(str(sticker_id))
    if not reply_text:
        reply_text = random.choice(["スタンプありがとうございます！🥰", "そのスタンプかわいいですね！", "お、いいスタンプ！私もほしいです！"])
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route("/")
def home():
    return "まこT LINE Bot is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
