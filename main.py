import queue
import time,os,sys
import random
from collections import deque
import rich
from rich.status import Status
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.style import Style
from rich.progress import Progress
import threading
from threading import *
from rich.live import Live
console = Console(log_path=False, log_time=True)

def clearConsole():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

clearConsole()


state = {
    "ket_quaround":"Đang Chờ",
    "win":0,
    "round":0,
    "lai":0,
    "lose":0,
    "win_round": 0,
    "lose_round": 0,
    "win_streak": 0,
    "lose_streak2": 0,
    "lose_streak3":0,
    "lose_streak4": 0,
    "tien_cuoc": 0,
    "skip": 0,
    "continue": 0,
    "color": None
}









def main():
    ensure_config_has_credentials()
    try:
        user_id = ensure_config_has_credentials()["userId"]
        secret_key = ensure_config_has_credentials()["secretKey"]
    except:
        console.log("lỗi Lấy thông tin")
        return main()

    console.log("[yellow] Chọn Loại Tiền Cược\n"
                  "[1] BUILD\n"
                  "[2] USDT\n"
                  "[3] WORLD")
    try:
        bet_type_input = int(console.input("[green] Mời Bạn Chọn (1/2/3): [/green]"))
        if bet_type_input == 1:
            bet_type = "BUILD"
            key_bet_type = "ctoken_contribute"
        elif bet_type_input == 2:
            bet_type = "USDT"
            key_bet_type = "ctoken_kusdt"
        elif bet_type_input == 3:
            bet_type = "WORLD"
            key_bet_type = "ctoken_kther"
    except ValueError:
        console.log("[white on red] Không đúng định dạng. Phải Nhập Số [/]")
        console.log("[blink]Đang Khởi động lại[/]")
        time.sleep(2)
        return main()

    wallet_guild = get_wallet(header_(user_id, secret_key))
    wl_ms = round(float(find_first_key(wallet_guild, key_bet_type)), 4)
    if key_bet_type == "ctoken_kusdt":
        bet_guild = round(float(wl_ms / 12 / 12 / 12), 4)
    elif key_bet_type == "ctoken_contribute":
        bet_guild = round(float(wl_ms / 12 / 12 / 12), 4)
    elif key_bet_type == "ctoken_kther":
        bet_guild = round(float(wl_ms / 12 / 12 / 12), 4)
    try:
        bet_amount = round(float(console.input(f"[blue] Nhập Số tiền cược (Theo Số dư của bạn khuyên {bet_guild} với hệ số 12: [/blue]")), 4)
    except Exception as e:
        console.log(f"[red] Lỗi: {e} \n"
                    f"[red] Đảm Bảo nhập đúng định dạng VD: 1 / 12.3 / 100 / 1000 không nhập chữ \n"
                    f"[red] Dùng Dấu chấm cho số thập phân, không dùng dấu phẩy [/]")
        return main()
    try:
        x_bet_amount = float(console.input("[cyan] Nhập Hệ Số Cược (Khuyên Để 12): [/cyan]"))
    except Exception as e:
        console.log(e)
    state["tien_cuoc"] = bet_amount
    console.log("[yellow] Chọn Thể Loại Chơi:\n"
                  "1. Đặt Theo LOGIC\n"
                  "2. Đặt Random\n"
                  "3. Đặt 1 Phòng cố định")
    try:
        chon_kieu = int(console.input("[yellow] Mời Bạn Lựa Chọn (1/2/3): [/yellow]"))
        if chon_kieu == 1:
            text = "Chọn Logic"
            chose_id = None
        elif chon_kieu == 2:
            text = "Random"
            chose_id = random.choice(range(1, 9))
        elif chon_kieu == 3:
            text = "Chọn 1 phòng cố định"
            console.log(f"[cyan] Mời bạn chọn Phòng cố định:\n"
                          f"1. {room_ID[1]}\n"
                          f"2. {room_ID[2]}\n"
                          f"3. {room_ID[3]}\n"
                          f"4. {room_ID[4]}\n"
                          f"5. {room_ID[5]}\n"
                          f"6. {room_ID[6]}\n"
                          f"7. {room_ID[7]}\n"
                          f"8. {room_ID[8]}")
            try:
                chose_id = int(console.input("[cyan] Mời bạn chọn:"))
            except ValueError:
                console.log("[white on red] Không đúng định dạng. Phải Nhập Số [/]")
                console.log("[blink]Đang Khởi động lại[/]")
                time.sleep(2)
                return main()

    except ValueError:
        console.log("[white on red] Không đúng định dạng. Phải Nhập Số [/]")
        console.log("[blink]Đang Khởi động lại[/]")
        time.sleep(2)
        return main()

    skip = console.input("Chơi Bao ván thì Bỏ Qua (999 hoặc enter để không dùng): ")

    if skip != 999:
        continue_ = console.input("Sau bao ván thì tiếp tục: ")
        if skip == "" or continue_ == "":
            console.log("Giá trị Skip và continue không đc để trống")
            return main()
        state["continue"] = continue_


    state["skip"] = int(skip)
    ######### hàm chạy phiên
    clearConsole()
    get_wl_guild_1 = get_wallet(header_(user_id, secret_key))
    wallet_build_1 = round(float(find_first_key(get_wl_guild_1, "ctoken_contribute")), 4)
    wallet_usdt_1 = round(float(find_first_key(get_wl_guild_1, "ctoken_kusdt")), 4)
    wallet_world_1 = round(float(find_first_key(get_wl_guild_1, "ctoken_kther")), 4)

    threading.Thread(target=header_thread, daemon=True).start()
    update_header(
        win_round=state["win_round"],
        ketquaround=state["ket_quaround"],
        bet_amount=bet_amount, x_bet_amount=x_bet_amount, chon_kieu=text,
        lai=state["lai"], lanthua=state["lose"], lose2=state["lose_streak2"], lose3=state["lose_streak3"], lose4=state["lose_streak4"],
        win=state["win"], round_=state["round"], win_streak=state["win_streak"],build=wallet_build_1,usdt=wallet_usdt_1,world=wallet_world_1, asset=bet_type
    )
    def run(wallet_build_1 = wallet_build_1, wallet_usdt_1 = wallet_usdt_1, wallet_world_1 = wallet_world_1):
        while True:
            data_ws = {
                "msg_type": "handle_enter_game",
                "asset_type": "BUILD",
                "user_id": int(user_id),
                "user_secret_key": secret_key
            }
            issue_id, second = run_second_ws(data_ws)
            if chon_kieu == 1:
                chose_id, _ = get_history(header_(user_id, secret_key), type_bet=bet_type)
            # Đặt Cược
            data_bet = {
                      "asset_type": bet_type,
                      "user_id": int(user_id),
                      "room_id": int(chose_id),
                      "bet_amount": round(float(state["tien_cuoc"]),4),
                    }

            if int(state["skip"]) > 0:
                if state["skip"] != 999 or state["skip"] != "":
                    state["skip"] -= 1
                    state["continue"] = int(continue_)
                cuoc = bet(header_(user_id, secret_key), data_bet)
                if cuoc["msg"] == "ok":
                    console.log(f"Kì->#{issue_id} -- [green] Đặt {state["tien_cuoc"]} Thành Công Vào {room_ID[chose_id]} [/]")
                    with Live(Panel("Chờ Sát Thủ"), refresh_per_second=4) as live:
                        for i in range(second + 8, 0, -1):
                            live.update(f"{i}")
                            time.sleep(1)

                    _, get_ket_qua = get_history(header_(user_id, secret_key), type_bet = bet_type)
                    if get_ket_qua != chose_id:
                        state["ket_quaround"] = "THẮNG"
                        state["round"] +=1
                        state["tien_cuoc"] = bet_amount
                        state["win_round"] += 1
                        if state["win_round"] >= state["win_streak"]:
                            state["win_streak"] += 1
                        state["win"] += 1
                        state["lose_round"] = 0
                        console.log(f"Kỳ->#{issue_id} -- [green] Thắng [/]  Sát thủ vào [yellow]{room_ID[get_ket_qua]} [/]")
                    else:
                        state["ket_quaround"] = "THUA"
                        state["round"] += 1
                        state["tien_cuoc"] = float(state["tien_cuoc"]) * float(x_bet_amount)
                        state["lose"] += 1
                        state["lose_round"] += 1
                        state["win_round"] = 0
                        if state["lose_round"] == 2:
                            state["lose_streak2"] += 1
                        elif state["lose_round"] == 3:
                            state["lose_streak3"] += 1
                        elif state["lose_round"] == 4:
                            state["lose_streak4"] += 1
                        console.log(f"Kỳ->#{issue_id} -- [red] Thua [/]  Sát thủ vào [yellow]{room_ID[get_ket_qua]} [/]")

                else:
                    if bet_type == "BUILD":
                        if state["tien_cuoc"] > wallet_build_1:
                            state["tien_cuoc"] = bet_amount
                    elif bet_type == "USDT":
                        if state["tien_cuoc"] > wallet_usdt_1:
                            state["tien_cuoc"] = bet_amount
                    elif bet_type == "WORLD":
                        if state["tien_cuoc"] > wallet_world_1:
                            state["tien_cuoc"] = bet_amount

                    console.log(f"Lỗi Đặt Cược: {cuoc["msg"]}")

            else:
                state["ket_quaround"] = "Bỏ Qua"
                if state["continue"] > 0:
                    state["continue"] -= 1
                console.log("Đang Bỏ Qua Ván Này")
                with Live(Panel("Chờ"), refresh_per_second=4) as live:
                    for i in range(second + 8, 0, -1):
                        live.update(f"{i}")
                        time.sleep(1)
                if state["continue"] == 0:
                    state["skip"] = int(skip)
                get_wl_guild = get_wallet(header_(user_id, secret_key))
                wallet_build = round(float(find_first_key(get_wl_guild, "ctoken_contribute")), 4)
                wallet_usdt = round(float(find_first_key(get_wl_guild, "ctoken_kusdt")), 4)
                wallet_world = round(float(find_first_key(get_wl_guild, "ctoken_kther")), 4)
                update_header(
                    win_round=state["win_round"],
                    ketquaround=state["ket_quaround"],
                              bet_amount=bet_amount, x_bet_amount=x_bet_amount, chon_kieu=text,
                              lai=state["lai"], lanthua=state["lose"], lose2=state["lose_streak2"],
                              lose3=state["lose_streak3"],
                              lose4=state["lose_streak4"],
                              win=state["win"], round_=state["round"], win_streak=state["win_streak"], build=wallet_build,
                              usdt=wallet_usdt, world=wallet_world, asset=bet_type)


            time.sleep(5)
            get_wl_guild = get_wallet(header_(user_id, secret_key))
            wallet_build = round(float(find_first_key(get_wl_guild, "ctoken_contribute")), 4)
            wallet_usdt = round(float(find_first_key(get_wl_guild, "ctoken_kusdt")), 4)
            wallet_world = round(float(find_first_key(get_wl_guild, "ctoken_kther")), 4)
            if bet_type == "BUILD":
                truoc = wallet_build_1
                tinh_lai = wallet_build - truoc
                wallet_build_1 = wallet_build
            elif bet_type == "USDT":
                truoc = wallet_usdt_1
                tinh_lai = wallet_usdt - truoc
                wallet_usdt_1 = wallet_usdt
            elif bet_type == "WORLD":
                truoc = wallet_world_1
                tinh_lai = wallet_world - truoc
                wallet_world_1 = wallet_world

            state["lai"] += round(float(tinh_lai), 4)
            if state["lai"] < 0:
                state["color"] = "[red]"
            elif state["lai"] > 0:
                state["color"] = "[green]"
            update_header(
                color_ = state["color"],
                win_round=state["win_round"],
                ketquaround=state["ket_quaround"],
                          bet_amount=bet_amount, x_bet_amount=x_bet_amount, chon_kieu=text,
                          lai=state["lai"], lanthua=state["lose"], lose2=state["lose_streak2"], lose3=state["lose_streak3"],
                          lose4=state["lose_streak4"],
                          win=state["win"], round_=state["round"], win_streak=state["win_streak"], build=wallet_build,
                          usdt=wallet_usdt, world=wallet_world, asset=bet_type)

    run()


if __name__ == "__main__":
    main()

