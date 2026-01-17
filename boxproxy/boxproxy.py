import os, json, uuid, datetime, socket
from aiohttp import web, ClientSession, ClientTimeout

UPSTREAM_BASE = os.environ.get("UPSTREAM_BASE", "http://api:8080").rstrip("/")
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "1048576"))  # 1 MiB default
DROPBOX_DIR_IN_CONTAINER="/dropbox"

os.makedirs(
    os.path.join(
        DROPBOX_DIR_IN_CONTAINER,
        socket.gethostname()
        ),
    exist_ok=True
)

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade"
}

def clean_headers(headers):
    out = {}
    for k, v in headers.items():
        if k.lower() not in HOP_BY_HOP:
            out[k] = v
    return out

async def handler(request: web.Request) -> web.StreamResponse:
    # Read and cap body
    body = await request.read()
    truncated = False
    if len(body) > MAX_BODY_BYTES:
        body = body[:MAX_BODY_BYTES]
        truncated = True

    # Forward upstream
    upstream_url = f"{UPSTREAM_BASE}{request.rel_url}"
    timeout = ClientTimeout(total=60)
    async with ClientSession(timeout=timeout) as session:
        async with session.request(
            method=request.method,
            url=upstream_url,
            params=request.query,
            data=body if body else None,
            headers=clean_headers(request.headers),
            allow_redirects=False,
        ) as resp:
            resp_body = await resp.read()

            paths_to_log = ["/v2/memories"]

            entry = None

            # Log (only if JSON-ish body; tweak as you like)
            ct = request.headers.get("content-type", "")
            request_path = request.path
            is_json = "application/json" in ct or ct.endswith("+json")
            if request_path in paths_to_log and is_json and body:
                ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
                rid = request.headers.get("x-request-id", str(uuid.uuid4()))
                try:
                    entry = json.loads(body.decode("utf-8", errors="replace"))
                except Exception:
                    # Not valid JSON despite content-type
                    entry = body.decode("utf-8", errors="replace")

                # Log each JSON request 
                if entry is not None:
                    filename = os.path.join(
                        DROPBOX_DIR_IN_CONTAINER,
                        socket.gethostname(),
                        f"{ts}_{resp.status}_{rid}.json"
                    )
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(entry["req_body"], f, ensure_ascii=False)

            # Return upstream response
            out = web.Response(body=resp_body, status=resp.status)
            for k, v in clean_headers(resp.headers).items():
                # Avoid setting invalid content-length; aiohttp handles it
                if k.lower() != "content-length":
                    out.headers[k] = v
            return out

app = web.Application(client_max_size=MAX_BODY_BYTES + 1024)
app.router.add_route("*", "/{tail:.*}", handler)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8080)
