# Face processing topology

`FaceDatabase` và `l2_distance` được định nghĩa trong `utils.py`.

## 1. Tạo feature database từ ảnh đăng ký

Một người có thể có nhiều ảnh đăng ký. Nên gom ảnh theo thư mục tên người; mỗi ảnh nên có đúng một khuôn mặt.

```text
db/
├── an/
│   └── an.jpg
└── tu/
    └── tu.jpg
```

```text
Mỗi ảnh trong db/<ten_nguoi>/
│
├── FaceDetector tìm mặt
├── Đúng một mặt? Nếu không thì báo lỗi, không thêm feature
├── FaceAlign(frame, bbox, landmarks) → mặt 112×112
├── FaceEmbedder → embedding đã chuẩn hóa, 512 chiều
└── Thêm embedding vào danh sách của người đó
    └── Lưu tất cả vào db/features.json
```

`db/features.json` ánh xạ mỗi người tới danh sách embedding. Mỗi ảnh tạo một embedding 512 giá trị float:

```text
"an": [embedding_01, embedding_02, ...]
```

Ảnh gốc vẫn nằm trong thư mục tên người. Có thể thêm nhiều ảnh vào cùng thư mục; mỗi ảnh sẽ thêm một embedding cho người đó. Gọi `FaceDatabase.build(...)` để tạo lại `db/features.json` sau khi thay đổi ảnh đăng ký, ví dụ:

```python
database = FaceDatabase("db/features.json")
database.build("db", detector, aligner, embedder)
```

## 2. Nhận diện một frame

```text
Input: 1 frame + db/features.json
│
├── FaceDetector
│   ├── Giữ detection có score ≥ 0.5
│   ├── NMS IoU threshold = 0.4
│   └── Danh sách khuôn mặt: bbox + 5 landmarks
│
├── Với mỗi khuôn mặt
│   │
│   ├── FaceAnti
│   │   ├── Crop frame gốc theo bbox, scale 2.7 → MiniFASNetV2
│   │   ├── Crop frame gốc theo bbox, scale 4.0 → MiniFASNetV1SE
│   │   └── Kết quả: REAL hoặc FAKE
│   │
│   ├── FAKE → gắn nhãn FAKE
│   │
│   └── REAL
│       ├── FaceAlign(frame, bbox, landmarks) → mặt 112×112
│       ├── FaceEmbedder → embedding đã chuẩn hóa
│       ├── Tính L2 với tất cả embedding trong database
│       ├── Lấy khoảng cách nhỏ nhất của từng người
│       ├── Khoảng cách tốt nhất đạt ngưỡng → gắn tên người
│       └── Không đạt ngưỡng → gắn nhãn UNKNOWN
│
└── Output: frame đã vẽ bbox và nhãn cho từng khuôn mặt
```

Nếu không thấy mặt thì giữ nguyên frame. Anti-spoof dùng crop theo bbox; embedding dùng mặt đã căn chỉnh theo 5 landmarks.
Pipeline mẫu trong `tests/test_face_onnx.py` nạp `db/features.json`; gọi `FaceDatabase.build(...)` khi cần cập nhật ảnh đăng ký.

Ngưỡng hiện tại: detect `0.5`, NMS IoU `0.4`, anti-spoof `0.5`, L2 match `≤ 1.15` (giá trị thử nghiệm). Nên đăng ký số ảnh tương đương cho mỗi người.
