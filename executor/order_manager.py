import time
import requests
import hmac
import hashlib

# Bybit REST 接口
BASE_URL = "https://api.bybit.com"

# 建议改为读取配置或环境变量
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"

def _sign(params):
    ordered = "&".join([f"{k}={params[k]}" for k in sorted(params)])
    return hmac.new(API_SECRET.encode("utf-8"), ordered.encode("utf-8"), hashlib.sha256).hexdigest()

def get_active_orders(symbol):
    """获取当前活动订单列表"""
    endpoint = "/v2/private/order/list"
    url = BASE_URL + endpoint
    timestamp = str(int(time.time() * 1000))
    params = {
        "api_key": API_KEY,
        "symbol": symbol,
        "timestamp": timestamp
    }
    params["sign"] = _sign(params)

    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        return res.json().get("result", {}).get("data", [])
    except Exception as e:
        print(f"❌ 获取订单失败: {e}")
        return []

def cancel_order(order_id, symbol):
    """撤销指定订单"""
    endpoint = "/v2/private/order/cancel"
    url = BASE_URL + endpoint
    timestamp = str(int(time.time() * 1000))
    params = {
        "api_key": API_KEY,
        "order_id": order_id,
        "symbol": symbol,
        "timestamp": timestamp
    }
    params["sign"] = _sign(params)

    try:
        res = requests.post(url, data=params, timeout=10)
        res.raise_for_status()
        print(f"🚫 已撤单：{order_id}")
    except Exception as e:
        print(f"❌ 撤单失败: {e}")

def monitor_orders(symbol, timeout_sec=30):
    """监控订单状态，必要时撤单"""
    print(f"🔍 正在检查 {symbol} 的订单状态...")
    orders = get_active_orders(symbol)

    for order in orders:
        status = order.get("order_status")
        order_id = order.get("order_id")
        created_time = int(order.get("created_time", 0)) // 1000

        if status == "New":
            age = time.time() - created_time
            if age > timeout_sec:
                print(f"⏱️ 订单已存在 {int(age)} 秒未成交，执行撤单：{order_id}")
                cancel_order(order_id, symbol)
            else:
                print(f"⌛ 订单尚未成交，等待中：{order_id}（{int(age)}秒）")
        else:
            print(f"✅ 已成交或已撤销订单：{order_id} - 状态：{status}")
