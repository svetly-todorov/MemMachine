#!/usr/bin/env bash
set -euo pipefail

KC_BASE="https://console.3.236.146.56.sslip.io/auth"
REALM="memmachine-platform"
CLIENT_ID="memmachine-platform-cli"          # <-- your new Keycloak client
SCOPE="openid profile email"        # add roles/offline_access if you need it

# 1) Start tunnel (you said you have this)
# TUNNEL_ID="$(start_cloudflare_tunnel_and_print_id)"
TUNNEL_ID="example-tunnel-id"

# 2) Ask Keycloak for a device code
device_json="$(
  curl -k -sS -X POST \
    "$KC_BASE/realms/$REALM/protocol/openid-connect/auth/device" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    --data-urlencode "client_id=$CLIENT_ID" \
    --data-urlencode "scope=$SCOPE"
)"

echo "DEBUG: device_json $device_json"

device_code="$(printf '%s' "$device_json" | python3 -c 'import sys,json; print(json.load(sys.stdin)["device_code"])')"
user_code="$(printf '%s' "$device_json" | python3 -c 'import sys,json; print(json.load(sys.stdin)["user_code"])')"
verify_uri="$(printf '%s' "$device_json" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("verification_uri_complete") or json.load(sys.stdin)["verification_uri"])' 2>/dev/null || true)"
interval="$(printf '%s' "$device_json" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("interval", 5))')"
expires_in="$(printf '%s' "$device_json" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("expires_in", 600))')"

echo ""
echo "Login required."
echo "Open: ${verify_uri:-$KC_BASE}"
echo "Enter code: $user_code"
echo ""

# 3) Poll token endpoint until authorized
deadline=$(( $(date +%s) + expires_in ))
access_token=""

while [ "$(date +%s)" -lt "$deadline" ]; do
  token_json="$(
    curl -k -sS -X POST \
      "$KC_BASE/realms/$REALM/protocol/openid-connect/token" \
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

echo "DEBUG: got access token $access_token"

# 4) Call your SaaS API to register the tunnel
# API_BASE="https://console.dev.memmachine.ai/api"   # <-- adjust to your real API host
# curl -sS -X POST \
#   "$API_BASE/tunnels/register" \
#   -H "Authorization: Bearer $access_token" \
#   -H "Content-Type: application/json" \
#   -d "{\"tunnel_id\":\"$TUNNEL_ID\"}" | cat

echo "Done."
