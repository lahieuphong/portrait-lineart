# run_lineart.py
import os
import sys
import subprocess

# chỉnh đường dẫn ảnh đầu vào cho đúng
INPUT = "data/input/HoChiMinhCityPostOffice.png"

# output mặc định sẽ là data/output/<basename>.png (nếu bạn muốn tên khác, sửa dòng SAVE_OUT)
basename = os.path.splitext(os.path.basename(INPUT))[0]
SAVE_OUT = os.path.join("data", "output", f"{basename}.png")
os.makedirs(os.path.dirname(SAVE_OUT), exist_ok=True)

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
    "--save_out", SAVE_OUT,        # SAVE: lưu PNG kết quả vào đây
    # nếu bạn muốn cửa sổ tự đóng sau khi vẽ -> thêm "--no_keep"
]

print("Running:", " ".join(cmd))
subprocess.run(cmd)
print("Done.")
