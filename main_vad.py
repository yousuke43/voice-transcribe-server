# main_vad.py (おしゃべりデバッグモード版)

import asyncio
import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# --- 1. 初期設定 (変更なし) ---
print("サーバーの初期設定を開始します...")
app = FastAPI()
print("Whisperモデルをロード中...")
model_size = "base"
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("Whisperモデルのロード完了。")
print("Silero VADモデルをロード中...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("Silero VADモデルのロード完了。")

# --- 2. WebSocket通信のメイン処理 ---
@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("クライアントが接続しました！")
    
    vad_iterator = VADIterator(vad_model, threshold=0.5)
    audio_buffer = bytearray()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)
            
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            speech_dict = vad_iterator(torch.from_numpy(audio_float32), return_seconds=True)
            
            # ★★★★★ 追加 ★★★★★
            # VADが何かを検知するたびに、その内容を表示する
            if speech_dict:
                print(f"VAD says: {speech_dict}")

            if speech_dict and 'end' in speech_dict:
                print(f"発話終了を検出。文字起こしを実行します...")

                full_audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16).copy()
                full_audio_float32 = full_audio_int16.astype(np.float32) / 32768.0
                
                segments, info = model.transcribe(full_audio_float32, beam_size=5, language="ja")
                
                transcription = "".join([s.text for s in segments])
                print(f"文字起こし結果: {transcription}")
                
                if transcription:
                    await websocket.send_text(transcription)
                
                audio_buffer.clear()
                vad_iterator.reset_states()

                # ★★★★★ 追加 ★★★★★
                # リセットが完了したことを明確に表示
                print("--- リセット完了、次の発話を待機中 ---")
                
    except WebSocketDisconnect:
        print("クライアントが切断しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        vad_iterator.reset_states()