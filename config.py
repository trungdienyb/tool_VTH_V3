import os
import json
import tempfile
from urllib.parse import urlparse, parse_qs, unquote

CONFIG_TXT = "config.txt"
CONFIG_JSON = "config.json"


def _atomic_write(path: str, content: str):

    dirn = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirn, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _parse_credentials_from_url(url: str):

    if not url or not isinstance(url, str):
        return {}
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return {}
    qs = parse_qs(parsed.query)
    user = None
    secret = None
    for k in ("userId", "userid", "user_id", "uid", "id"):
        if k in qs and qs[k]:
            user = qs[k][0]
            break
    for k in ("secretKey", "secretkey", "secret_key", "key", "token"):
        if k in qs and qs[k]:
            secret = qs[k][0]
            break
    if user:
        user = unquote(user).strip()
    if secret:
        secret = unquote(secret).strip()
    if user and secret:
        return {"userId": str(user), "secretKey": str(secret)}
    return {}


def _valid_credentials(creds: dict) -> bool:

    if not creds:
        return False
    user = creds.get("userId")
    sk = creds.get("secretKey")
    if not (isinstance(user, str) and isinstance(sk, str)):
        return False
    ulen = len(user.strip())
    sklen = len(sk.strip())
    return (6 <= ulen <= 7) and (sklen == 64)


def ensure_config_has_credentials(
    txt_path: str = CONFIG_TXT,
    json_path: str = CONFIG_JSON,
    prompt_message: str = None,
) -> dict:

    if prompt_message is None:
        prompt_message = (
            "Dán URL chứa userId & secretKey (vd: https://.../?userId=...&secretKey=...) "
            "hoặc nhập 'userId secretKey' (cách nhau 1 dấu cách): "
        )


    candidate = None
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    # lấy dòng đầu
                    candidate = content.splitlines()[0].strip()
        except Exception:
            candidate = None

    if candidate:
        creds = _parse_credentials_from_url(candidate)
        if _valid_credentials(creds):
            # lưu json và trả về
            _atomic_write(json_path, json.dumps(creds, ensure_ascii=False, indent=2))
            return creds
        # nếu parse được nhưng không hợp lệ -> tiếp xuống bước nhập

    # 2) vòng lặp yêu cầu nhập cho tới khi hợp lệ
    while True:
        user_input = input(prompt_message).strip()
        if not user_input:
            print("Không được để trống. Thử lại.")
            continue

        # nếu input chứa '=' hoặc bắt đầu bằng http -> coi như URL
        if "=" in user_input or user_input.lower().startswith("http"):
            parsed = _parse_credentials_from_url(user_input)
            if parsed and _valid_credentials(parsed):
                # lưu URL gốc vào config.txt, lưu json
                save_url = user_input if user_input.endswith("\n") else user_input + "\n"
                _atomic_write(txt_path, save_url)
                _atomic_write(json_path, json.dumps(parsed, ensure_ascii=False, indent=2))
                print("Đã lưu config từ URL.")
                return parsed
            else:
                print("Không tìm được userId/secretKey hợp lệ trong URL (userId 6-7 ký tự, secretKey 64 ký tự). Thử lại.")
                continue

        # nếu input 2 phần phân cách space -> coi là userId secretKey
        parts = user_input.split()
        if len(parts) == 2:
            user_try, secret_try = parts[0].strip(), parts[1].strip()
            candidate_creds = {"userId": user_try, "secretKey": secret_try}
            if _valid_credentials(candidate_creds):
                # lưu dạng URL vào config.txt để tương thích
                fake_url = f"https://escapemaster.net/battleroyale/?userId={user_try}&secretKey={secret_try}"
                _atomic_write(txt_path, fake_url + "\n")
                _atomic_write(json_path, json.dumps(candidate_creds, ensure_ascii=False, indent=2))
                print("Đã lưu config.")
                return candidate_creds
            else:
                print("userId hoặc secretKey không hợp lệ (userId 6-7 ký tự, secretKey 64 ký tự). Thử lại.")
                continue

        # input không hợp lệ
        print("Đầu vào không hợp lệ. Dán URL hoặc nhập 'userId secretKey'. Thử lại.")

print(ensure_config_has_credentials()["userId"])
print(ensure_config_has_credentials()["secretKey"])