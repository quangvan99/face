# BÀI TOÁN 1.1: SO KHỚP KHUÔN MẶT THỰC TẾ (CAMERA) VỚI ẢNH TRÊN CCCD (1:1 VERIFICATION)

| Mã TC | Tên kịch bản | Dữ liệu đầu vào (CCCD & Camera) | Các bước thực hiện | Kết quả kỳ vọng |
| :--- | :--- | :--- | :--- | :--- |
| **TC-1.1.01** | Chính chủ quẹt thẻ và đứng trước camera | Ảnh chip CCCD người A;<br>Camera bắt mặt trực tiếp người A | 1. Quẹt thẻ CCCD lấy ảnh gốc.<br>2. Camera chụp mặt người A.<br>3. Trích xuất ArcFace 512D cả 2 ảnh.<br>4. Tính Cosine Similarity. | **Trạng thái MATCH** (Khớp danh tính). Hệ thống ghi nhận đúng chính chủ. |
| **TC-1.1.02** | Người khác dùng thẻ CCCD người khác (Mượn thẻ) | Ảnh chip CCCD người A;<br>Camera bắt mặt người B | 1. Quẹt thẻ người A.<br>2. Người B đứng trước camera.<br>3. So khớp khuôn mặt 2 nguồn. | **Trạng thái MISMATCH**. Từ chối xác thực, hiển thị cảnh báo người không khớp thẻ. |
| **TC-1.1.03** | Chính chủ đeo kính thuốc gọng mỏng / đổi kiểu tóc | Ảnh chip CCCD người A (không kính);<br>Người A đeo kính thuốc đứng trước camera | 1. Đọc thẻ CCCD.<br>2. Camera chụp người A đeo kính.<br>3. Trích xuất đặc trưng & so khớp. | **Trạng thái MATCH**. Landmark và vùng mắt/mũi/miệng vẫn nhận diện chuẩn xác. |
| **TC-1.1.04** | Góc nghiêng khuôn mặt (Yaw / Pitch +- 15 đến 30 độ) | Ảnh chip CCCD người A;<br>Người A đứng nghiêng nhẹ trước camera | 1. Đọc thẻ CCCD.<br>2. Người A quay mặt nghiêng góc 15-30 độ.<br>3. FaceAligner xoay chuẩn hóa 112x112. | **Trạng thái MATCH**. Hệ thống tự động căn chỉnh góc và so khớp thành công. |
| **TC-1.1.05** | Khoảng cách xa, kích thước mặt < 120x120 px | Ảnh CCCD người A;<br>Người A đứng xa camera (mặt ~80x80 px) | 1. Đọc thẻ CCCD.<br>2. Camera phát hiện mặt nhưng bbox < 120 px.<br>3. Kiểm tra điều kiện kích thước. | Hệ thống từ chối so khớp, hiển thị thông báo *"Yêu cầu đứng gần camera hơn"*. |
| **TC-1.1.06** | Khoảng cách chuẩn, kích thước mặt >= 120x120 px | Ảnh CCCD người A;<br>Người A đứng đúng vạch quầy check-in | 1. Đọc thẻ CCCD.<br>2. Camera phát hiện mặt bbox >= 120 px.<br>3. Chạy full pipeline so khớp. | Đủ điều kiện tiêu chuẩn Bảng 5, so khớp chính xác với độ tương đồng cao. |
| **TC-1.1.07** | Thử nghiệm nghiệm thu hàng loạt (>= 20 người) | 20 cặp dữ liệu gồm ảnh CCCD và mặt camera của 20 cán bộ Ialy | 1. Chạy kiểm thử cho 20 cặp dữ liệu.<br>2. Đánh giá tỷ lệ chính xác. | Tỷ lệ nhận dạng chính xác đạt từ **90% đến 100%** (tối thiểu 18/20 cặp chính xác). |

## Fixture bổ sung để thử nhanh


| File | Biến thể | Có thể dùng để thử |
| :--- | :--- | :--- |
| `tests/fixtures/cam_an_deokinh.jpg` | Ảnh camera của An đeo kính | TC-1.1.03 |
| `cam_real_roll_-15.jpg`, `cam_real_roll_+15.jpg` | Xoay toàn khung hình ±15° | Độ bền với ảnh bị xoay trong mặt phẳng |
| `cam_real_roll_-30.jpg`, `cam_real_roll_+30.jpg` | Xoay toàn khung hình ±30° | Xoay trong mặt phẳng ở góc lớn hơn |
| `cam_real_bright.jpg`, `cam_real_dim.jpg` | Tăng sáng hoặc giảm sáng/tương phản | Điều kiện sáng khác nhau |
| `cam_real_face_180px.jpg`, `cam_real_face_100px.jpg` | Thu nhỏ toàn cảnh để khuôn mặt xấp xỉ kích thước mục tiêu | Thử mặt đủ lớn và mặt nhỏ/xa |
| `cam_real_yaw_left.png`, `cam_real_yaw_right.png` | Ảnh AI chỉnh đầu quay trái/phải | Smoke test yaw |
| `cam_real_pitch_up.png`, `cam_real_pitch_down.png` | Ảnh AI chỉnh đầu ngẩng/cúi | Smoke test pitch |

Có thể ghép từng ảnh camera biến thể với `an_cccd.jpg` để thử MATCH và `hien_cccd.jpg` để thử MISMATCH. Các ảnh này là dữ liệu mô phỏng; ảnh AI có thể làm thay đổi nhẹ đặc điểm khuôn mặt và không thay thế mẫu thu thật. Chúng bổ sung smoke test, không thay thế nghiệm thu thực tế.

Chạy các kiểm tra 1:1 trên bộ fixture hiện có:

```bash
pytest tests/test_face_1_1.py
```

Ngưỡng cosine `0.5` trong test chỉ dùng để phân biệt rõ các fixture hiện có; cần hiệu chỉnh lại trên tập xác thực thực tế trước khi dùng vận hành.

# BÀI TOÁN 1.2: PHÁT HIỆN GIẢ MẠO KHUÔN MẶT (FACE ANTI-SPOOFING / LIVENESS)

| Mã TC | Tên kịch bản | Dữ liệu đầu vào (CCCD & Camera) | Các bước thực hiện | Kết quả kỳ vọng |
| :--- | :--- | :--- | :--- | :--- |
| **TC-1.2.01** | Người thật trực tiếp trước camera (Live check) | Người thật đứng trước camera quầy/cửa, biểu cảm tự nhiên | 1. Camera bắt khuôn mặt.<br>2. Cắt crop tỷ lệ 2.7x và 4.0x.<br>3. MiniFASNet ensemble dự đoán liveness. | Phân loại nhãn **REAL / LIVE**. Khung xanh. Cho phép chuyển sang bước so khớp. |
| **TC-1.2.02** | Giả mạo bằng ảnh in giấy A4 màu (Print Attack) | Tấm ảnh in màu chất lượng cao chân dung người A cầm giơ trước camera | 1. Đưa ảnh in trước camera thay mặt người.<br>2. Hệ thống quét bề mặt và viền ảnh.<br>3. Trích xuất liveness score. | Phân loại **SPOOF** (Giả mạo ảnh in). Khung đỏ, khóa quy trình xác thực, cảnh báo. |
| **TC-1.2.03** | Giả mạo bằng ảnh in đen trắng / giấy photo | Tấm ảnh photo đen trắng chân dung người A | 1. Đưa ảnh giấy photo trước camera.<br>2. Hệ thống phân tích vi hạt ảnh và phản xạ giấy. | Phân loại **SPOOF**. Không vượt qua bước kiểm tra người thật. |
| **TC-1.2.04** | Giả mạo bằng màn hình smartphone (Replay Attack) | Màn hình iPhone/Android hiển thị ảnh chân dung tĩnh của người trong CCCD | 1. Cầm điện thoại mở ảnh giơ trước camera.<br>2. Hệ thống bắt vân sọc Moire và viền bezel điện thoại. | Phân loại **SPOOF** (Giả mạo màn hình). Chặn ngay lập tức. |
| **TC-1.2.05** | Giả mạo bằng video phát trên iPad/Tablet | Video người thật đang nói chuyện, chớp mắt phát trên màn hình tablet | 1. Phát video trên tablet giơ trước camera.<br>2. Mô hình MiniFASNet 4.0x bắt phẳng không gian và viền tablet. | Phân loại **SPOOF** (Giả mạo video phát lại). Cảnh báo xâm nhập. |
| **TC-1.2.06** | Giả mạo bằng chính thẻ CCCD giơ trước camera | Cầm thẻ CCCD thật đưa sát ống kính camera để camera đọc ảnh thẻ | 1. Đưa thẻ CCCD vào vùng quét camera.<br>2. Hệ thống kiểm tra chất liệu nhựa và kích thước ảnh. | Phân loại **SPOOF / Vật thể phi sinh học**. Yêu cầu người thật nhìn vào camera. |
| **TC-1.2.07** | Thử nghiệm nghiệm thu Anti-spoof hàng loạt | Tập dữ liệu >= 20 mẫu người thật + >= 20 mẫu giả mạo các loại | 1. Chạy nghiệm thu hàng loạt trên tập dữ liệu.<br>2. Đánh giá tỷ lệ chính xác tổng hợp. | Độ chính xác tổng hợp đạt **>= 90%** (chuẩn Bảng 5 yêu cầu thử >= 20 người). |

Fixture anti-spoof hiện có:

| File | Kịch bản gần nhất | Ghi chú |
| :--- | :--- | :--- |
| `tests/fixtures/cam_real.jpg` | TC-1.2.01 | Ảnh camera người thật An |
| `tests/fixtures/cam_fake.jpg` | TC-1.2.04 | Một frame có mặt trên màn hình tablet; chưa kiểm tra được nội dung video động |
| `tests/fixtures/augmented_1_2/cam_print_color.png` | TC-1.2.02 | Ảnh tổng hợp An được in màu trên giấy A4 |
| `tests/fixtures/augmented_1_2/cam_print_bw.png` | TC-1.2.03 | Ảnh tổng hợp An dạng photocopy đen trắng |
| `tests/fixtures/augmented_1_2/cam_card_demo.png` | TC-1.2.06 | Thẻ ảnh demo có nhãn DEMO, không phải CCCD thật |

Chạy smoke test: `pytest tests/test_face_1_2.py`. Ảnh in và thẻ là dữ liệu tổng hợp, chỉ dùng để kiểm tra sơ bộ model trên các fixture này.
