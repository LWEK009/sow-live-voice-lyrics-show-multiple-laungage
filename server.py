"""
Sow Live — live speech-to-text web UI (Moonshine Voice backend).

Maintainer / source hub: https://github.com/LWEK009

  cd ~/moonshine-web-live
  source ~/.venvs/moonshine/bin/activate
  pip install -r requirements.txt
  uvicorn server:app --host 127.0.0.1 --port 8765

Open http://127.0.0.1:8765 — microphone needs HTTPS except on localhost.
"""

from __future__ import annotations

import asyncio
import json
import struct
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from moonshine_voice import get_model_for_language
from moonshine_voice.transcriber import (
    LineCompleted,
    LineStarted,
    LineTextChanged,
    Stream,
    Transcriber,
    TranscriptEventListener,
)

GITHUB_USERNAME = "LWEK009"
GITHUB_PROFILE_URL = f"https://github.com/{GITHUB_USERNAME}"

STATIC = Path(__file__).resolve().parent / "static"

# One transcriber per language (heavy); each WebSocket gets its own Stream.
_transcribers: dict[str, Transcriber] = {}
_locks: dict[str, asyncio.Lock] = {}


async def _get_transcriber(language: str) -> Transcriber:
    key = language.lower().strip() or "en"
    if key in _transcribers:
        return _transcribers[key]
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        if key not in _transcribers:
            path, arch = await asyncio.to_thread(
                get_model_for_language, key, None
            )
            _transcribers[key] = Transcriber(path, arch, update_interval=0.35)
    return _transcribers[key]


class _WsListener(TranscriptEventListener):
    def __init__(self, queue: asyncio.Queue) -> None:
        super().__init__()
        self._q = queue

    def _put(self, payload: dict) -> None:
        try:
            self._q.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    def on_line_started(self, event: LineStarted) -> None:
        self._put(
            {
                "type": "line_started",
                "lineId": str(event.line.line_id),
                "text": event.line.text or "",
            }
        )

    def on_line_text_changed(self, event: LineTextChanged) -> None:
        self._put(
            {
                "type": "partial",
                "lineId": str(event.line.line_id),
                "text": event.line.text or "",
                "complete": bool(event.line.is_complete),
            }
        )

    def on_line_completed(self, event: LineCompleted) -> None:
        self._put(
            {
                "type": "final",
                "lineId": str(event.line.line_id),
                "text": event.line.text or "",
            }
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for t in _transcribers.values():
        try:
            t.close()
        except Exception:
            pass
    _transcribers.clear()


app = FastAPI(
    title="Sow Live",
    description=f"Live voice transcription — GitHub: {GITHUB_PROFILE_URL}",
    lifespan=lifespan,
)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.websocket("/ws")
async def websocket_transcribe(websocket: WebSocket, lang: str = "en"):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    stream: Optional[Stream] = None
    sender = asyncio.create_task(_drain_queue_to_ws(websocket, queue))

    try:
        await websocket.send_json({"type": "status", "message": "loading_model"})
        transcriber = await _get_transcriber(lang)
        stream = transcriber.create_stream(update_interval=0.35)
        listener = _WsListener(queue)
        stream.add_listener(listener)
        stream.start()
        await websocket.send_json({"type": "status", "message": "ready"})

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue
            raw = message.get("bytes")
            if raw is not None:
                if len(raw) < 8:
                    continue
                sample_rate, n_floats = struct.unpack_from("<II", raw, 0)
                payload = raw[8:]
                if len(payload) != n_floats * 4:
                    continue
                floats = list(struct.unpack(f"<{n_floats}f", payload))
                if floats:
                    stream.add_audio(floats, int(sample_rate))
            elif message.get("text") is not None:
                try:
                    cmd = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if cmd.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()
        try:
            await sender
        except asyncio.CancelledError:
            pass
        if stream is not None:
            try:
                stream.remove_all_listeners()
                stream.stop()
                stream.close()
            except Exception:
                pass


async def _drain_queue_to_ws(websocket: WebSocket, queue: asyncio.Queue) -> None:
    try:
        while True:
            item = await queue.get()
            try:
                await websocket.send_json(item)
            except Exception:
                break
    except asyncio.CancelledError:
        raise
