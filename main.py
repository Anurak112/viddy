"""
Main FastAPI application for the ssstik-clone backend.

This service exposes endpoints to download TikTok content as MP3 or MP4, to fetch stories,
and to enqueue those downloads as background jobs. Each request is protected by a rate
limiter and optional API key authentication. Requests may also include a Cloudflare
Turnstile token when the relevant environment variables are configured. When running
inside Docker Compose the Redis service is used both for rate limiting and the job
queue.
"""

import os
from typing import Optional

import redis
from fastapi import FastAPI, Request, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask
import yt_dlp
from rq import Queue
import uuid
import aiofiles
import httpx


# Configure Redis connection. The default points at the local Redis instance.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = redis.from_url(REDIS_URL)

# Create a dedicated queue for download jobs. Jobs will persist results in Redis
# until fetched by the client.
job_queue = Queue("downloads", connection=redis_conn)

app = FastAPI(title="ssstik-clone backend")

# Parse a list of API keys from the environment. If no keys are provided then
# authentication is effectively disabled. Keys may be supplied either via the
# query string (api_key) or in the X-API-Key header.
API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]

# Determine the per-minute request limit. This limit applies individually to each
# API key (if provided) or otherwise to the requester's IP address. The value
# should be a small integer; defaults to 20.
RATE_LIMIT = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MIN", "20"))

async def rate_limit_dependency(request: Request) -> None:
    """Simple token bucket rate limiter using Redis.

    Each request increments a counter associated with the caller's identifier. If the
    counter exceeds RATE_LIMIT within a 60 second window then a 429 error is raised.
    The identifier is either the API key (if supplied) or the remote IP address.
    """
    identifier = request.headers.get("X-API-Key") or request.query_params.get("api_key") or request.client.host
    key = f"rate-limit:{identifier}"
    current = redis_conn.get(key)
    if current is None:
        # First request: create a key with expiry of 60s.
        redis_conn.setex(key, 60, 1)
    else:
        count = int(current)
        if count >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Too many requests; please slow down.")
        redis_conn.incr(key)


async def api_key_dependency(request: Request) -> None:
    """Enforce API key checking when keys are configured.

    If API_KEYS contains one or more values then every request must include an
    accepted key either in the X-API-Key header or the api_key query parameter. If
    API_KEYS is empty then this dependency does nothing.
    """
    if not API_KEYS:
        return
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


async def turnstile_dependency(request: Request) -> None:
    """Verify a Cloudflare Turnstile token when configured.

    If TURNSTILE_SECRET_KEY is set in the environment then each request must send
    a 'cf-turnstile-response' token via the query string or in the headers. The
    token is verified using Cloudflare's verification endpoint. When the secret
    key is not configured this dependency is a no-op.
    """
    secret_key = os.getenv("TURNSTILE_SECRET_KEY")
    if not secret_key:
        return
    token = request.headers.get("cf-turnstile-response") or request.query_params.get("cf-turnstile-response")
    if not token:
        raise HTTPException(status_code=400, detail="Missing Turnstile token.")
    # Compose verification request. Remote IP is passed to tighten validation.
    verify_url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    payload = {
        "secret": secret_key,
        "response": token,
        "remoteip": request.client.host,
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(verify_url, data=payload, timeout=10.0)
            data = resp.json()
            if not data.get("success"):
                raise HTTPException(status_code=400, detail="Turnstile verification failed.")
        except Exception as exc:
            # If verification fails due to network errors or invalid responses we
            # treat it as a failure. This prevents bypassing the challenge by
            # disrupting external requests.
            raise HTTPException(status_code=400, detail="Turnstile verification error.") from exc


def _download_mp3(url: str) -> dict:
    """Blocking helper to download and convert a TikTok to MP3.

    Uses yt_dlp to extract the best audio stream and convert it to an MP3 using
    FFmpeg. Returns a mapping with the file path on disk and a suggested name.
    """
    job_id = uuid.uuid4().hex
    outtmpl = f"/tmp/{job_id}.%(ext)s"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "noprogress": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    mp3_path = outtmpl.replace("%(ext)s", "mp3")
    title = info.get("title", "audio")
    return {"file": mp3_path, "name": f"{title}.mp3"}


def _download_mp4(url: str) -> dict:
    """Blocking helper to download a TikTok to MP4 with no watermark when possible.

    Extracts the best available video and audio streams and merges them into a
    single MP4 file. The output filename is based off a unique identifier.
    """
    job_id = uuid.uuid4().hex
    outtmpl = f"/tmp/{job_id}.mp4"
    ydl_opts = {
        "format": "bv*+ba/best",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    title = info.get("title", "video")
    return {"file": outtmpl, "name": f"{title}.mp4"}


def _download_story(url: str) -> dict:
    """Blocking helper for story downloads. Uses the same logic as MP4 downloads."""
    return _download_mp4(url)


async def _stream_file_and_cleanup(path: str, filename: str, media_type: str) -> StreamingResponse:
    """Create a streaming response for a file and ensure cleanup after sending.

    Files produced by yt_dlp are stored on disk in /tmp. They should be removed
    after they've been streamed to the client to avoid leaking disk space. The
    background task cleans up the file once the response is complete.
    """
    async def file_iterator():
        async with aiofiles.open(path, mode="rb") as f:
            chunk = await f.read(1024 * 1024)
            while chunk:
                yield chunk
                chunk = await f.read(1024 * 1024)
    # Schedule file removal after response has been sent
    async def cleanup():
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    headers = {"Content-Disposition": f"attachment; filename=\"{filename}\""}
    return StreamingResponse(file_iterator(), media_type=media_type, headers=headers, background=BackgroundTask(cleanup))


@app.get("/api/download-mp3")
async def download_mp3(
    url: str,
    queue_mode: Optional[bool] = Query(False),
    request: Request = None,
    _: None = Depends(rate_limit_dependency),
    __: None = Depends(api_key_dependency),
    ___: None = Depends(turnstile_dependency),
):
    """Download a TikTok video as MP3 or enqueue it as a background job.

    When `queue_mode` is true the request enqueues a job and returns a job ID. The
    client should poll the `/api/jobs/{id}` endpoint until the job finishes.
    Otherwise the download happens synchronously and the MP3 is streamed back to
    the client immediately.
    """
    if queue_mode:
        job = job_queue.enqueue(_download_mp3, url)
        return {"job_id": job.get_id()}
    # Run the blocking operation off the event loop
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _download_mp3, url)
    return await _stream_file_and_cleanup(result["file"], result["name"], "audio/mpeg")


@app.get("/api/download-mp4")
async def download_mp4(
    url: str,
    queue_mode: Optional[bool] = Query(False),
    request: Request = None,
    _: None = Depends(rate_limit_dependency),
    __: None = Depends(api_key_dependency),
    ___: None = Depends(turnstile_dependency),
):
    """Download a TikTok video as MP4 or enqueue it as a background job."""
    if queue_mode:
        job = job_queue.enqueue(_download_mp4, url)
        return {"job_id": job.get_id()}
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _download_mp4, url)
    return await _stream_file_and_cleanup(result["file"], result["name"], "video/mp4")


@app.get("/api/download-story")
async def download_story(
    url: str,
    queue_mode: Optional[bool] = Query(False),
    request: Request = None,
    _: None = Depends(rate_limit_dependency),
    __: None = Depends(api_key_dependency),
    ___: None = Depends(turnstile_dependency),
):
    """Download a TikTok story or enqueue it as a background job."""
    if queue_mode:
        job = job_queue.enqueue(_download_story, url)
        return {"job_id": job.get_id()}
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _download_story, url)
    return await _stream_file_and_cleanup(result["file"], result["name"], "video/mp4")


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Return the status of a previously enqueued job.

    If the job has finished then the result is included. Otherwise the status is
    reported as either `queued` or `failed`. If the job cannot be found a 404
    response is returned.
    """
    job = job_queue.fetch_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.is_finished:
        return {"status": "finished", "result": job.result}
    if job.is_failed:
        return {"status": "failed"}
    return {"status": "queued"}


@app.get("/api/jobs/{job_id}/download")
async def download_completed_job(job_id: str):
    """Download the output of a completed job.

    This endpoint should be called only after `GET /api/jobs/{job_id}` reports a
    finished status. It streams the file result and removes it from disk once
    complete.
    """
    job = job_queue.fetch_job(job_id)
    if job is None or not job.is_finished:
        raise HTTPException(status_code=404, detail="Job not ready")
    result = job.result
    return await _stream_file_and_cleanup(result["file"], result["name"], "application/octet-stream")