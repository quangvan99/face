# Phương án chạy Face Recognition trên NPU RK3588

## Kết luận

Các model ONNX hiện tại có thể được thử triển khai trên NPU RK3588, nhưng luồng phù hợp là chuyển từng model từ ONNX sang định dạng `.rknn` bằng RKNN-Toolkit2, sau đó chạy trên board bằng RKNN Lite2 (Python) hoặc RKNN Runtime (C/C++). RKNN-Toolkit2 hỗ trợ RK3588; bản mới nhất được công bố trong tài liệu đang tham khảo là 2.3.2. [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)

Không nên chỉ đổi `providers` của ONNX Runtime để mong model chạy trên NPU. ONNX Runtime RKNPU Execution Provider hiện ghi nền tảng hỗ trợ là RK1808 Linux, không phải RK3588. [RKNPU Execution Provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/RKNPU-ExecutionProvider.html)

## Trạng thái board và code hiện tại

- Board hiện tại là Firefly AIO-3588JD4, SoC RK3588; `/usr/lib/librknnrt.so` đã tồn tại.
- ONNX Runtime 1.23.2 trên board báo providers `AzureExecutionProvider` và `CPUExecutionProvider`, chưa có RKNPU provider.
- Các session trong `face_onnx.py` đang chỉ định `CPUExecutionProvider`, nên inference hiện chạy trên CPU.
- Đã thấy thư viện runtime, nhưng chưa xác định đầy đủ phiên bản driver NPU và runtime để xác nhận chúng tương thích với bộ Toolkit dùng chuyển model.

## Luồng triển khai đề xuất

Giữ nguyên các model hiện tại trước, chuyển riêng từng ONNX thành RKNN:

| Module | Model | Hướng triển khai |
|---|---|---|
| Detect | `scrfd_10g_bnkps.onnx`, input 640×640 | Thử chuyển model hiện tại; giữ decode bbox, NMS và landmarks ở CPU. |
| Anti-spoof | `MiniFASNetV2.onnx`, `MiniFASNetV1SE.onnx`, input 80×80 | Chuyển cả hai model; giữ ensemble và cách gộp score hiện tại. |
| Align | `FaceAlign` | Giữ ở CPU; căn chỉnh ảnh từ 5 landmarks. |
| Embedding | `arcface_r100.onnx`, input 112×112 | Chuyển model R100; xác nhận embedding và kết quả nhận diện trước khi quyết định lượng tử hóa. |

```text
Video
  → CPU: decode frame, resize
  → NPU: SCRFD
  → CPU: decode bbox/landmarks, NMS
  → CPU: FaceAlign
  → NPU: MiniFASNetV2 + MiniFASNetV1SE
  → nếu là mặt thật: NPU ArcFace R100
  → CPU: so embedding với DB, vẽ và ghi video
```

NPU tăng tốc phần suy luận model. Decode video, tiền xử lý ảnh, alignment, NMS, so embedding và vẽ video vẫn nằm trên CPU.

RKNN Model Zoo có ví dụ RetinaFace chuyển ONNX sang RKNN và chạy trên RK3588. Đây là phương án dự phòng nếu SCRFD hiện tại không chuyển được; thay detector sẽ cần cập nhật bộ giải mã output và kiểm tra lại độ chính xác. [Ví dụ RetinaFace trên RK3588](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/RetinaFace/README.md)

## Kế hoạch độ chính xác và tốc độ

1. Tạo bản `fp` chưa lượng tử hóa cho từng model để làm baseline so sánh với ONNX chạy CPU. RKNN Model Zoo dùng `fp` cho chế độ không lượng tử hóa và `i8` cho INT8. [Hướng dẫn chuyển RetinaFace sang RKNN](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/RetinaFace/README.md)
2. Dùng cùng ảnh và cùng tiền xử lý để so kết quả ONNX với RKNN: bbox và landmarks của detector; score thật/giả của anti-spoof; embedding và quyết định tìm danh tính của ArcFace.
3. Sau khi baseline đúng, thử INT8 riêng từng model với tập calibration đại diện cho camera thực tế: điều kiện sáng, góc mặt, khoảng cách và các dạng ảnh giả thường gặp.
4. Nếu INT8 làm giảm độ chính xác, dùng mixed/hybrid precision cho các layer nhạy cảm hoặc giữ model đó ở `fp`. Tài liệu Rockchip đề xuất hybrid quantization hoặc QAT khi INT8 làm giảm chất lượng. [RKNN Model Zoo FAQ](https://github.com/airockchip/rknn_model_zoo/blob/main/FAQ_CN.md) · [Accuracy analysis và hybrid quantization](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/functions/accuracy_analysis)
5. Đánh giá lại các ngưỡng sau khi đổi runtime hoặc lượng tử hóa. Đặc biệt, không mặc định ngưỡng L2 `1.15` vẫn phù hợp với embedding mới; kiểm tra lại false accept/false reject trên cặp ảnh cùng người và khác người. Anti-spoof hiện dùng `0.5`; cũng cần kiểm tra lại trên ảnh thật và ảnh giả.

ArcFace R100 là model lớn nhất trong bộ hiện tại. Nên đo bản `fp` trên NPU trước, rồi mới so tốc độ và độ chính xác của INT8/mixed precision. Không lượng tử hóa đồng loạt cả ba module nếu chưa đánh giá kết quả theo tác vụ.

## Runtime và benchmark

- Bắt đầu bằng RKNN Lite2 để giữ phần tích hợp Python gọn. Nếu cần tối ưu thông lượng sau khi đo end-to-end, chuyển phần gọi inference sang RKNN C API; FAQ của Rockchip lưu ý Python API có thể chậm hơn.
- Đo cả pipeline trên board, gồm decode, tiền xử lý, inference, hậu xử lý và ghi video. Thời gian `rknn.run` riêng không đại diện cho tốc độ toàn ứng dụng; xung CPU/NPU/DDR và tải hệ thống cũng ảnh hưởng kết quả. [RKNN Model Zoo FAQ](https://github.com/airockchip/rknn_model_zoo/blob/main/FAQ_CN.md)
- RK3588 hỗ trợ chọn NPU core mask. Bắt đầu với cấu hình mặc định rồi benchmark các cấu hình core trên board; không giả định dùng nhiều core sẽ luôn nhanh hơn, nhất là khi các bước detect → anti-spoof → embedding phụ thuộc kết quả bước trước. [RK3588 core mask API](https://github.com/airockchip/rknpu2/blob/master/runtime/RK3588/Linux/librknn_api/include/rknn_api.h)
- Trước khi triển khai, xác nhận phiên bản NPU driver, `librknnrt` và RKNN Lite2 tương thích với nhau theo SDK/firmware của Firefly.

## Tài liệu tham khảo

- [RKNN-Toolkit2: nền tảng hỗ trợ, API triển khai và phiên bản](https://github.com/airockchip/rknn-toolkit2)
- [RKNN Model Zoo: ví dụ RetinaFace và chuyển ONNX sang RKNN cho RK3588](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/RetinaFace/README.md)
- [ONNX Runtime RKNPU EP: nền tảng hỗ trợ](https://onnxruntime.ai/docs/execution-providers/community-maintained/RKNPU-ExecutionProvider.html)
- [RKNN Model Zoo FAQ: hiệu năng Python/C API, benchmark và lượng tử hóa](https://github.com/airockchip/rknn_model_zoo/blob/main/FAQ_CN.md)
