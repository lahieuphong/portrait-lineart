#

# 📘 Portrait Lineart Generator (Turtle Version)

Script chuyển ảnh chân dung thành **lineart dạng nét vẽ** bằng Turtle Graphics + xử lý ảnh, sau đó **xuất ra file PNG** chất lượng cao.

Dự án gồm 2 phần chính:

- `run_lineart.py` — file chạy nhanh, tiện sử dụng
- `src/portrait_lineart_turtle.py` — toàn bộ logic xử lý & vẽ lineart

#

## ⚙️ Yêu cầu

### Python

- Python **3.8 – 3.12**

### Thư viện cần cài

Cài đặt bằng:

```bash
pip install -r requirements.txt
```

Hoặc nếu không có file requirements, chạy:

```bash
pip install pillow numpy matplotlib scipy
```

Không cần Ghostscript vì hệ thống **xuất PNG trực tiếp bằng PIL**.

---

## 📁 Cấu trúc thư mục

```
project/
│
├── run_lineart.py
├── src/
│   └── portrait_lineart_turtle.py
│
├── data/
│   ├── input/
│   │   └── your_image.png
│   └── output/
│       └── generated_lineart.png
│
└── README.md
```

---

## ▶️ Cách chạy

1. Copy ảnh gốc vào thư mục:

```
data/input/
```

2. Mở `run_lineart.py` và chỉnh đường dẫn INPUT nếu cần:

```python
INPUT = "data/input/your_image.png"
```

3. Chạy:

```bash
python run_lineart.py
```

4. Kết quả sẽ được lưu tự động vào:

```
data/output/<tên-ảnh>.png
```

---

## 🛠️ Các tùy chọn quan trọng trong `run_lineart.py`

| Tham số                     | Ý nghĩa                               |
| --------------------------- | ------------------------------------- |
| `--fast`                    | Vẽ nhanh hơn (giảm cập nhật màn hình) |
| `--chaikin 2`               | Mức làm mượt đường cong               |
| `--cr_samples 8`            | Tăng độ mượt khi dùng Catmull-Rom     |
| `--line_width 1`            | Nét cơ bản                            |
| `--thickness_mode length`   | Nét dài → dày hơn                     |
| `--min_width / --max_width` | Giới hạn độ dày của nét               |
| `--eps`                     | Ngưỡng cạnh (0.5–1.0 là tốt)          |
| `--blur`                    | Làm mờ ảnh trước khi lấy line         |
| `--edge_mul`                | Tăng bắt nét                          |
| `--batch 50`                | Vẽ nhóm 50 nét một lần để nhanh hơn   |
| `--save_out`                | Đường dẫn file PNG đầu ra             |
| `--no_keep`                 | Không giữ cửa sổ lại sau khi vẽ       |

---

## 📤 Về việc xuất ảnh (PNG)

Hệ thống **không còn sử dụng PostScript (.ps)**.
Thay vào đó, các nét được vẽ lại lên một `PIL.Image` và lưu trực tiếp thành:

```
PNG chuẩn 24-bit, xem được trên mọi hệ thống
```

Điều này giúp tránh lỗi:

```
image.png.20251203_xxxxx.ps
```

---

## 🎯 Mục tiêu

- Chuyển ảnh thành lineart chất lượng cao
- Giữ nét mềm, mượt, tự nhiên
- Cho phép tùy chỉnh độ nhạy, độ dày nét, mức làm mượt
- Có cửa sổ Turtle để xem trực tiếp quá trình vẽ
- Xuất PNG để dùng cho illustration, in ấn, hoặc stylized rendering

---

## 📌 Ghi chú

- Ảnh càng rõ mặt → lineart càng đẹp
- Kích thước ảnh khoảng 512–1024px là tối ưu
- Nếu muốn chạy nhanh hơn: thêm `--fast` hoặc giảm `--batch` xuống 30
- Nếu ảnh quá nhiễu: tăng `--min_path_len` lên 10–15

---
