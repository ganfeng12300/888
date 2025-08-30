def check_result(result):
    if not result or "status" not in result:
        raise ValueError("❌ 订单返回异常，无成交状态字段")
    if result["status"].lower() != "filled":
        print(f"⚠️ 警告：订单状态为 {result['status']}，可能未完全成交")
