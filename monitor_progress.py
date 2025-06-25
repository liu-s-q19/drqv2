import os
import time
import re

LOG_DIR = "exp_local/2025.06.25"
REFRESH_INTERVAL = 3  # 刷新间隔（秒）
LINES_TO_SHOW = 3     # 每个任务显示日志行数

def extract_tqdm_line(lines):
    # 匹配 tqdm 进度条的典型格式
    for line in reversed(lines):
        if re.search(r'Training.*\|.*\[.*\]', line):
            return line
    return None

def tail(filename, n=10):
    """读取文件最后n行"""
    with open(filename, 'rb') as f:
        f.seek(0, os.SEEK_END)
        filesize = f.tell()
        size = 1024
        data = b''
        while filesize > 0 and data.count(b'\n') <= n:
            seek = max(0, filesize - size)
            f.seek(seek)
            data = f.read(filesize - seek) + data
            filesize = seek
        return data.decode(errors='ignore').splitlines()[-n:]

def main():
    while True:
        os.system('clear')
        print("==== 任务进度监控 ====")
        log_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.log')])
        for log_file in log_files:
            print(f"\n{log_file}:")
            log_path = os.path.join(LOG_DIR, log_file)
            try:
                lines = tail(log_path, n=LINES_TO_SHOW)
                tqdm_line = extract_tqdm_line(lines)
                if tqdm_line:
                    print(f"进度: {tqdm_line}")
                else:
                    for line in lines:
                        print(line)
            except Exception as e:
                print(f"读取失败: {e}")
        print("\n按 Ctrl+C 退出监控")
        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    main()