"""Send preprocessed OpenBMI recorded trials to the GUI; no EEG hardware required."""
import argparse
import socket
import time
from receiver import load_trials, send_trial

def main():
    parser = argparse.ArgumentParser(description="OpenBMI MAT 文件网络回放发送端")
    parser.add_argument("--data", required=True, help="原始 OpenBMI MAT 文件")
    parser.add_argument("--phase", choices=["train", "test"], default="test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    parser.add_argument("--interval", type=float, default=1, help="试次之间的等待秒数")
    args = parser.parse_args()
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"等待 GUI 连接 {args.host}:{args.port}", flush=True)
        connection, address = server.accept()
        with connection:
            print(f"已连接 {address}，正在预处理记录……", flush=True)
            for item in load_trials(args.data, args.phase):
                send_trial(connection, item)
                print(f"已发送 {item[0]}/{item[1]}", flush=True)
                time.sleep(max(0, args.interval))

if __name__ == "__main__":
    main()

