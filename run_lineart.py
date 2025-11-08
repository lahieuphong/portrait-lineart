# run_lineart.py
import os
import sys
import subprocess

# chỉnh đường dẫn ảnh đầu vào cho đúng
INPUT = "data/input/picture_1.jpg"

cmd = [
    sys.executable, "src/portrait_lineart_turtle.py",
    "-i", INPUT,
    "--fast",
    "--chaikin", "1",
    "--cr_samples", "6",
    "--line_width", "2.5",
    "--thickness_mode", "fixed",
    # NOTE: đã bỏ --output_svg để không lưu file tự động
]

print("Running:", " ".join(cmd))
subprocess.run(cmd)
print("Done.")
