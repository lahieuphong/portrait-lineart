# run_lineart.py
import os
import sys
import subprocess

# chỉnh đường dẫn ảnh đầu vào cho đúng
INPUT = "data/input/69836bb0-c90e-4a83-85c1-7e12017ee7cf.png"

cmd = [
    sys.executable, "src/portrait_lineart_turtle.py",
    "-i", INPUT,
    "--fast",                      # Bật chế độ vẽ nhanh
    "--chaikin", "2",              # Làm mượt vừa phải
    "--cr_samples", "8",           # Tăng độ mượt đường cong
    "--line_width", "1",           # Nét mảnh
    "--thickness_mode", "length",  # Nét dài dày hơn
    "--min_width", "0.2",          # Nét mảnh tối thiểu
    "--max_width", "1.2",          # Nét dày tối đa
    "--eps", "0.8",                # Giữ chi tiết
    "--blur", "0.5",               # Giảm làm mờ
    "--edge_mul", "0.8",           # Bắt nhiều cạnh hơn
    "--min_path_len", "5",         # Bỏ nhiễu
    "--batch", "50",               # Vẽ 50 nét rồi mới cập nhật màn hình
]

print("Running:", " ".join(cmd))
subprocess.run(cmd)
print("Done.")