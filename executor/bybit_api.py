import requests
import time
import hmac
import hashlib

# ✅ 请替换为您的真实 API Key / Secret（务必放入环境变量或配置文件中）
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
BASE_URL = "https://api.bybit.com"

def _sign(params):
    ordered = "&".join([f"{k}={params[k]}" for k in sorted(params)])
    return hmac.new(
        API_SECRET.encode("utf-8"),
        ordered.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

def place_order(symbol, side, qty, strategy=None):
    endpoint = "/v2/private/order/create"
    url = BASE_URL + endpoint

    timestamp = str(int(time.time() * 1000))
    params = {
        "api_key": API_KEY,
        "symbol": symbol,
        "side": side.upper(),
        "order_type": "Market",
        "qty": qty,
        "time_in_force": "GoodTillCancel",
        "timestamp": timestamp
    }

    params["sign"] = _sign(params)

    try:
        response = requests.post(url, data=params, timeout=10)
        response.raise_for_status()
        return response.json()["result"]
    except Exception as e:
        print(f"❌ API请求异常: {e}")
        raise
