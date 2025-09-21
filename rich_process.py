import threading, time, queue
from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()
state_q: "queue.Queue[dict]" = queue.Queue()


def render_header(
    color_,
    win_round: int,
    ketquaround:str,
    bet_amount: float,
    x_bet_amount: int,
    chon_kieu: int,
    lai: float,
    lanthua: int,
    lose2: int,
    lose3: int,
    lose4: int,
    win: int,
    round_: int,
    win_streak: int,
    build:float,
    usdt:float,
    world:float,
    asset: str = "BUILD"
):
    content = (
        f"             Phiên Bản Hiện Tại (VTH - V3.1)\n\n"
        f"Loại:            {chon_kieu}\n"
        f"Mức Cược:        {bet_amount} {asset}\n"
        f"Hệ Số:           {x_bet_amount}\n"
        f"Lãi:             {color_}{round(float(lai), 4)} {asset} [/]\n"
        f"Số Trận Thắng:   {win}/{round_}\n"
        f"Chuỗi Thắng:     {win_round} | Max: {win_streak}\n"
        f"Lần Thua:        {lanthua}\n"
        f"Chuỗi Thua 2:    {lose2}\n"
        f"Chuỗi Thua 3:    {lose3}\n"
        f"Chuỗi Thua 4:    {lose4}\n\n\n"
        f"                Số Dư\n\n"
        f"        BUILD:           {build}\n"
        f"        WORLD:           {world}\n"
        f"        USDT:            {usdt}\n\n"
        f"      Kết Quả Trận Vừa Rồi:    {'[green]' + ketquaround if ketquaround == "THẮNG" else '[red]' + ketquaround}\n"
    )
    return Panel(
        content,
        title="Cấu Hình Người Chơi",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2)
    )


def header_thread():
    """Luồng luôn in header ở đầu màn hình."""
    while True:
        state = state_q.get()  # chờ state mới
        # Di chuyển con trỏ lên đầu, xóa màn hình
        print("\033[H\033[J", end="")
        # In lại header
        console.print(render_header(**state))


def update_header(**kwargs):
    """Hàm tiện để update state"""
    state_q.put(kwargs)