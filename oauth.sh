#!/usr/bin/env bash
set -euo pipefail

CONSOLE_BASE="https://console.3.236.146.56.sslip.io/"
KEYCLOAK_API="${CONSOLE_BASE}/auth"
PLATFORM_API="${CONSOLE_BASE}/api"
REALM="memmachine-platform"
CLIENT_ID="memmachine-platform-cli"          # <-- your new Keycloak client
SCOPE="openid profile email"        # add roles/offline_access if you need it

# 1) Check if user has already provided the cloudflare tunnel URL as an environment variable
TUNNEL_URL=""
if test -z ${CLOUDFLARE_TUNNEL_URL:-}; then
  echo "CLOUDFLARE_TUNNEL_URL is not set."
  read -p "Enter the Cloudflare Tunnel URL for this host: " TUNNEL_URL
else
  TUNNEL_URL=${CLOUDFLARE_TUNNEL_URL}
fi

# 2) Ask Keycloak for a device code
device_json="$(
  curl -k -sS -X POST \
    "$KEYCLOAK_API/realms/$REALM/protocol/openid-connect/auth/device" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    --data-urlencode "client_id=$CLIENT_ID" \
    --data-urlencode "scope=$SCOPE"
)"

device_code="$(printf '%s' "$device_json" | python3 -c 'import sys,json; print(json.load(sys.stdin)["device_code"])')"
user_code="$(printf '%s' "$device_json" | python3 -c 'import sys,json; print(json.load(sys.stdin)["user_code"])')"
verify_uri="$(printf '%s' "$device_json" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("verification_uri_complete") or json.load(sys.stdin)["verification_uri"])' 2>/dev/null || true)"
interval="$(printf '%s' "$device_json" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("interval", 5))')"
expires_in="$(printf '%s' "$device_json" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("expires_in", 600))')"

echo ""
echo "Login required."
echo "Open ${verify_uri:-$KEYCLOAK_API} and log in to the MemMachine Platform Console."
echo "If prompted for a code, enter: $user_code"
echo ""

# 3) Poll token endpoint until authorized
deadline=$(( $(date +%s) + expires_in ))
access_token=""

while [ "$(date +%s)" -lt "$deadline" ]; do
  token_json="$(
    curl -k -sS -X POST \
      "$KEYCLOAK_API/realms/$REALM/protocol/openid-connect/token" \
      -H "Content-Type: application/x-www-form-urlencoded" \
      --data-urlencode "grant_type=urn:ietf:params:oauth:grant-type:device_code" \
      --data-urlencode "client_id=$CLIENT_ID" \
      --data-urlencode "device_code=$device_code"
  )"

  # If success, it will contain access_token; if not, it contains error fields
  access_token="$(printf '%s' "$token_json" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("access_token",""))' 2>/dev/null || true)"
  if [ -n "$access_token" ]; then
    break
  fi

  err="$(printf '%s' "$token_json" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("error",""))' 2>/dev/null || true)"
  case "$err" in
    authorization_pending)
      sleep "$interval"
      ;;
    slow_down)
      interval=$((interval + 2))
      sleep "$interval"
      ;;
    expired_token|access_denied)
      echo "Login failed: $err"
      exit 1
      ;;
    *)
      # Could be transient / unexpected
      sleep "$interval"
      ;;
  esac
done

if [ -z "$access_token" ]; then
  echo "Login timed out."
  exit 1
fi

# 4) Call your SaaS API to register the tunnel
curl -k --request POST \
  --url ${PLATFORM_API}/v0/membox/tunnels \
  --header 'Accept: application/json' \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${access_token}" \
  --data "{\"device_name\": \"$(hostname)\", \"public_url\": \"${TUNNEL_URL}\"}"

echo "Done."
