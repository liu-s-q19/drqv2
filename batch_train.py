import os
import subprocess
import time

TASK_DIR = "cfgs/task"
TRAIN_SCRIPT = "train.py"
EXP_DIR = "exp_local"  # 假设训练结果保存在这里
AGENT_NAME = "drqv2"    # 可切换为 "vsac"
GPU_LIST = ["0", "1"]
NUM_WORKERS = 2  # 最大并发数

def run_task(task_file, gpu_id):
    task_name = os.path.splitext(task_file)[0]
    exp_path = os.path.join(EXP_DIR, task_name)
    log_path = f"logs/{task_name}.log"
    os.makedirs("logs", exist_ok=True)
    # 检查是否已完成
    if os.path.exists(exp_path):
        print(f"已存在 {exp_path}，跳过 {task_file}")
        return None
    cmd = [
        "python", TRAIN_SCRIPT,
        f"+task={task_name}",
        f"agent={AGENT_NAME}"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"开始训练: {task_file}，分配到GPU {gpu_id}")
    logf = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=logf, stderr=logf, env=env)
    return (p, logf, task_file)

def main():
    task_files = [f for f in os.listdir(TASK_DIR) if f.endswith('.yaml')]
    running = []
    idx = 0
    for task_file in task_files:
        # 控制并发数
        while len([p for p, _, _ in running if p.poll() is None]) >= NUM_WORKERS:
            time.sleep(5)
            # 清理已结束的进程
            running = [(p, logf, t) for p, logf, t in running if p.poll() is None]
        gpu_id = GPU_LIST[idx % len(GPU_LIST)]
        result = run_task(task_file, gpu_id)
        if result:
            running.append(result)
        idx += 1
    # 等待所有进程结束
    for p, logf, t in running:
        p.wait()
        logf.close()
        print(f"任务完成: {t}")
    print("所有任务已完成。")

if __name__ == "__main__":
    main() 