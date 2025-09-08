import os, hmac, hashlib, time, asyncio, logging  
from typing import Optional

import httpx
from fastapi import APIRouter, Request, Header, HTTPException, FastAPI
from fastapi.responses import PlainTextResponse

from slack_service import SlackService

from dotenv import load_dotenv
load_dotenv()  
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() or "INFO"
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["Slack"])
slack_service = SlackService()

MEMORY_BACKEND_URL = os.getenv("MEMORY_BACKEND_URL", "http://localhost:8000")

@router.post("/events")
async def slack_events(
    request: Request,
    x_slack_signature: Optional[str] = Header(default=None, alias="X-Slack-Signature"),
    x_slack_request_timestamp: Optional[str] = Header(default=None, alias="X-Slack-Request-Timestamp"),
):
    signing_secret = os.getenv("SLACK_SIGNING_SECRET", "")
    raw_body: bytes = await request.body()

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if payload.get("type") == "url_verification":
        return PlainTextResponse(content=payload.get("challenge", ""))

    if not x_slack_signature or not x_slack_request_timestamp or \
       not verify_slack_signature(signing_secret, x_slack_request_timestamp, x_slack_signature, raw_body):
        logger.warning("[SLACK] Invalid signature")
        raise HTTPException(status_code=401, detail="Invalid Slack signature")

    if payload.get("type") != "event_callback":
        return PlainTextResponse(content="ignored")

    event = payload.get("event") or {}
    event_type = event.get("type")
    subtype = event.get("subtype")
    bot_id = event.get("bot_id")
    channel = event.get("channel")
    user = event.get("user")
    text = event.get("text") or ""
    ts = event.get("ts")
    thread_ts = event.get("thread_ts")

    logger.info(f"[SLACK] event={event_type} channel={channel} user={user} ts={ts} thread={thread_ts} text={text!r}")

    if event_type != "message" or subtype is not None or bot_id:
        return PlainTextResponse(content="ignored")

    # optional channel restrictions
    allow_channel = os.getenv("CRM_CHANNEL_ID")
    if allow_channel and channel != allow_channel:
        return PlainTextResponse(content="ignored")

    stripped = (text or "").lstrip()
    low = stripped.lower()

    if low.startswith("*q") or low.startswith("*q "):
        query_text = stripped[2:].lstrip()
        asyncio.create_task(process_query_and_reply(channel, ts, thread_ts, user, query_text))
        return PlainTextResponse(content="ok")
    
    else:
        asyncio.create_task(process_memory_post(channel, ts, thread_ts, user, text))
        return PlainTextResponse(content="ok")


def verify_slack_signature(secret: str, timestamp: str, signature: str, body: bytes) -> bool:
    try:
        req_ts = int(timestamp)
    except Exception:
        return False
    if abs(int(time.time()) - req_ts) > 300:
        return False

    base_string = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    computed = hmac.new(secret.encode("utf-8"), base_string, hashlib.sha256).hexdigest()
    expected = f"v0={computed}"
    return hmac.compare_digest(expected, signature)


async def process_memory_post(channel: str, ts: str, thread_ts: Optional[str], user: str, text: str):
    """Post all messages to memory system"""
    logger.info(f"[SLACK] process_memory_post start: channel={channel} user={user} text={text!r}")

    author_name = await slack_service.get_user_display_name(user) if user else (user or "")
    
    memory_url = f"{MEMORY_BACKEND_URL}/memory"
    
    params = {
        "user_id": user,
        "query": text
    }

    logger.info(f"[SLACK] POST -> {memory_url} params={params}")

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(memory_url, params=params)
            logger.info(f"[SLACK] memory post resp status={resp.status_code}")
            if resp.status_code == 200:
                logger.info("[SLACK] Message posted to memory successfully")
            else:
                logger.warning(f"[SLACK] Failed to post to memory: {resp.status_code}")
    except Exception as e:
        logger.error(f"[SLACK] Error posting to memory: {e}")


async def process_query_and_reply(channel: str, ts: str, thread_ts: Optional[str], user: str, query_text: str):
    """Handle *Q queries by searching memory and using OpenAI chat completion"""
    logger.info(f"[SLACK] process_query_and_reply start: channel={channel} user={user} query={query_text!r}")

    author_name = await slack_service.get_user_display_name(user) if user else (user or "")
    
    search_url = f"{MEMORY_BACKEND_URL}/memory"
    
    params = {
        "query": query_text,
        "user_id": user,
        "timestamp": str(int(time.time()))
    }

    logger.info(f"[SLACK] GET -> {search_url} params={params}")

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(search_url, params=params)
            logger.info(f"[SLACK] memory search resp status={resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    formatted_query = data.get("formatted_query", "")
                    
                    response_text = await generate_openai_response(formatted_query, query_text)
                else:
                    response_text = f"⚠️ Search failed: {data.get('message', 'Unknown error')}"
            else:
                response_text = f"⚠️ Search failed with status {resp.status_code}"
                
    except Exception as e:
        logger.error(f"[SLACK] Error searching memory: {e}")
        response_text = f"⚠️ Error searching memory: {str(e)}"

    await slack_service.post_message(channel=channel, text=response_text, thread_ts=thread_ts or ts)
    logger.info("[SLACK] posted query response")


async def generate_openai_response(formatted_query: str, original_query: str) -> str:
    """Generate response using OpenAI chat completion"""
    try:
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment")
            return "⚠️ OpenAI API key not configured"
        
        client = openai.AsyncOpenAI(api_key=api_key)
        
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful CRM assistant. Use the provided context to answer the user's question accurately and concisely."
            },
            {
                "role": "user",
                "content": formatted_query
            }
        ]
        
        logger.info(f"[OPENAI] Sending request with {len(formatted_query)} characters")
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"[OPENAI] Generated response: {len(response_text)} characters")
        
        return response_text
        
    except Exception as e:
        logger.error(f"[OPENAI] Error generating response: {e}")
        return f"⚠️ Error generating AI response: {str(e)}"


app = FastAPI(title="CRM Slack Integration")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)