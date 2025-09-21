from email import header
import requests, json, os
import time
from logic import *
import asyncio
import json
import websockets
# from logic import _load_state, _save_state

# Auto reconnect configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CONNECTION_TIMEOUT = 30  # seconds
READ_TIMEOUT = 30  # seconds



def auto_reconnect(func):
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
                print(f"Lỗi kết nối (lần thử {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Đang thử lại sau {RETRY_DELAY} giây...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("Đã hết số lần thử lại. Vui lòng kiểm tra kết nối mạng.")
                    raise e
            except Exception as e:
                print(f"Lỗi không xác định: {str(e)}")
                raise e

    return wrapper


def make_request_with_retry(method, url, headers=None, json_data=None, params=None):
    """Hàm thực hiện request với retry và timeout"""
    session = requests.Session()
    session.timeout = (CONNECTION_TIMEOUT, READ_TIMEOUT)

    for attempt in range(MAX_RETRIES):
        try:
            if method.upper() == 'GET':
                response = session.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                response = session.post(url, headers=headers, json=json_data)
            else:
                raise ValueError(f"Phương thức HTTP không được hỗ trợ: {method}")

            response.raise_for_status()
            return response

        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
            print(f"Lỗi kết nối (lần thử {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                print(f"Đang thử lại sau {RETRY_DELAY} giây...")
                time.sleep(RETRY_DELAY)
            else:
                print("Đã hết số lần thử lại. Vui lòng kiểm tra kết nối mạng.")
                raise e
        except Exception as e:
            print(f"Lỗi không xác định: {str(e)}")
            raise e


HOST_DOMAIN = ["user.3games.io",
               "api.escapemaster.net"]


# # #

def header_(user_id, secret_key):
    return {
        "user-secret-key": secret_key,
        "content-type":"application/json",
        "Accept-Language": "vi-VN",
        "user-id": str(user_id),
        "origin": "https://escapemaster.net",
        "referer": "https://escapemaster.net/",
    }




room_ID = {
    1: "Nhà Kho",
    2: "Phòng Họp",
    3: "Phòng Giám Đốc",
    4: "Phòng Trò Chuyện",
    5: "Phòng Giám Sát",
    6: "Văn Phòng",
    7: "Phòng Tài Vụ",
    8: "Phòng Nhân Sự"
}


def find_first_key(data, target_key):
    if isinstance(data, dict):
        for k, v in data.items():
            if k == target_key:
                return v
            result = find_first_key(v, target_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_first_key(item, target_key)
            if result is not None:
                return result
    return None
def find_all_keys(data, parent_key=""):
    keys = []

    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            keys.append(full_key)
            keys.extend(find_all_keys(v, full_key))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            full_key = f"{parent_key}[{i}]"
            keys.extend(find_all_keys(item, full_key))

    return keys

def check_connection(header, host=HOST_DOMAIN[0]):
    """Kiểm tra kết nối tới server"""
    try:
        response = make_request_with_retry('GET',
                                           f"https://{host}/user/regist?is_cwallet=1&is_mission_setting=true&version=&time={int(time.time())}",
                                           headers=header)
        if response.status_code == 200:
            print("Kết nối thành công!")
            return True
        else:
            print(f"Kết nối thất bại với mã lỗi: {response.status_code}")
            return False
    except Exception as e:
        print(f"Không thể kết nối: {str(e)}")
        return False


@auto_reconnect
def get_history(header, type_bet="BUILD", host=HOST_DOMAIN[1]):
    try:

        response = make_request_with_retry('GET', f"https://{host}/escape_game/recent_100_issues?asset={type_bet}",headers=header)
        data = response.json()
        response_10 = make_request_with_retry('GET', f"https://{host}/escape_game/recent_10_issues?asset={type_bet}", headers=header)
        data_10 = response_10.json()
        recent_100 = find_first_key(data, "room_id_2_killed_times")
        recent_10 = find_first_key(data_10, "data")
        chose_id = choose_safe_room(recent_100, recent_10)
        kill_room = find_first_key(data_10, "killed_room_id")
        return chose_id, kill_room

    except Exception as e:
        print(f"Lỗi trong get_history: {str(e)}")
        raise e


@auto_reconnect
def get_wallet(header, host=HOST_DOMAIN[0]):
    try:
        response = make_request_with_retry('GET',
                                           f"https://{host}/user/regist?is_cwallet=1&is_mission_setting=true&version=&time={int(time.time())}",
                                           headers=header)
        data = response.json()
        return data
    except Exception as e:
        print(f"Lỗi trong get_wallet: {str(e)}")
        raise e


@auto_reconnect
def bet(header, data, host=HOST_DOMAIN[1]):
    try:
        response = make_request_with_retry('POST', f"https://{host}/escape_game/bet", headers=header, json_data=data)
        data = response.json()
        return data
    except Exception as e:
        print(f"Lỗi trong bet: {str(e)}")
        raise e



async def second_ws(data):
    while True:  # vòng ngoài để chạy lại khi cần
        try:
            async with websockets.connect(
                "ws://api.escapemaster.net/escape_master/ws",
                ping_interval=20,     # gửi ping mỗi 20s
                ping_timeout=10       # nếu 10s không có pong thì coi như mất kết nối
            ) as ws:
                await ws.send(json.dumps(data))
                while True:
                    try:
                        # timeout cho mỗi lần chờ tin nhắn
                        reply = await asyncio.wait_for(ws.recv(), timeout=20)
                        data = json.loads(reply)
                        msg = data.get("msg_type")
                        if msg == "notify_count_down":
                            issue_id = data.get("issue_id")
                            secon = data.get("count_down")
                            return issue_id, secon # hoặc xử lý rồi break
                    except asyncio.TimeoutError:
                        print("Timeout: no message received, reconnecting...")
                        await asyncio.sleep(3)
                        return  second_ws
        except Exception as e:
            print("Reconnect after error:", e)
            await asyncio.sleep(3)  # chờ rồi kết nối lại


def run_second_ws(data):
    return asyncio.run(second_ws(data))
#
# user_id = 2329625
# secret_key = "6eb86e1dbd612c523506378cc53f6e7c91a0cebaa12a561d55a825a5baec87f4"
# data_ws = {
#         "msg_type": "handle_enter_game",
#         "asset_type": "BUILD",
#         "user_id": int(user_id),
#         "user_secret_key":secret_key,
#             }
# print(run_second_ws(data_ws))

