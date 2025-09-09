# main_vad.py (おしゃべりデバッグモード版)
import asyncio
from collections import deque

import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# --- 1. FastAPI インスタンス作成 ---
app = FastAPI()
print("FastAPI サーバーを初期化しました。")

# --- 2. Whisperモデルロード ---
print("Whisperモデルをロード中...")
model_size = "base"
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("Whisperモデルのロード完了。")

# --- 3. Silero VADモデルロード ---
print("Silero VADモデルをロード中...")
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("Silero VADモデルのロード完了。")

# --- 4. WebSocket エンドポイント ---
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("クライアントが接続しました！")
    
    vad_iterator = VADIterator(vad_model, threshold=0.5)
    audio_buffer = bytearray()

    llm_wating = False  # LLM応答待ちフラグ
    waiting_messages = deque()  # LLM応答待ちメッセージキュー

    messages=[
        {"role": "system", "content": "あなたは親切で有能なアシスタントです。"},
    ]

    # --- 4-1. sendToLLM関数（内部定義） ---
    async def sendToLLM() :
        nonlocal llm_wating, waiting_messages
        latest_reply = ""
        while waiting_messages:
            message=waiting_messages.popleft()
            messages.append({"role": "user", "content": message})
            print(f"LLMにメッセージを送信: {messages}")
            await asyncio.sleep(3)
            reply = {f"role": "assistant", "content": {message+"OK"} }
            messages.append(reply)
            print(f"LLMからの応答: {messages}")

            latest_reply = reply
        
        if latest_reply:
            await websocket.send_text(reply["content"])
            print(f"クライアントに応答を送信: {messages}")
        
        llm_wating = False
        print("LLM応答待ちフラグをリセット。")
      

    # --- 4-2. 音声受信ループ ---
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            # NumPy 配列に変換
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # VADで発話判定
            speech_dict = vad_iterator(torch.from_numpy(audio_float32), return_seconds=True)
            if speech_dict:
                print(f"VAD says: {speech_dict}")

            # 発話終了を検出した場合
            if speech_dict and 'end' in speech_dict:
                print("発話終了を検出。文字起こしを実行します...")

                full_audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16).copy()
                full_audio_float32 = full_audio_int16.astype(np.float32) / 32768.0
                
                # Whisperで文字起こし
                segments, info = model.transcribe(full_audio_float32, beam_size=5, language="ja")
                transcription = "".join([s.text for s in segments])
                print(f"文字起こし結果: {transcription}")

                if transcription:
                    if not llm_wating:
                        llm_wating = True
                        waiting_messages.append(transcription)

                        asyncio.create_task(sendToLLM())
                    else:
                        waiting_messages.append(transcription)
                        print(waiting_messages)
                        print(f"LLM応答待ちのため、メッセージをキューに追加: {transcription}")
                
                # バッファとVAD状態をリセット
                audio_buffer.clear()
                vad_iterator.reset_states()
                print("--- リセット完了、次の発話を待機中 ---")
                
    except WebSocketDisconnect:
        print("クライアントが切断しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        vad_iterator.reset_states()
