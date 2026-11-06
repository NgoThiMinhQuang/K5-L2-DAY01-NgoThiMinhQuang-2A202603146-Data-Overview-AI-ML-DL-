# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

---

### Từ cần nhớ

| Từ | Nghĩa ngắn |
|---|---|
| Prediction | Kết quả mô hình dự đoán, chưa phải nhãn đúng |
| Ground truth | Nhãn chuẩn được xác lập theo guideline |
| Taxonomy | Danh sách lớp được định nghĩa sẵn, ví dụ ImageNet-1K, COCO-80 |
| Checkpoint | File mô hình `.pt` đã được huấn luyện |
| Guideline | Quy tắc gán nhãn dữ liệu |
| Annotator | Người thực hiện gán nhãn |
| Reviewer / QC | Người kiểm tra chất lượng nhãn |
| Escalation | Chuyển trường hợp khó lên Lab Coach để xử lý |
| Model score | Điểm dự đoán của mô hình, không phải độ chính xác của nhãn |

---

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.

Chỉ sử dụng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

### 1.1. Record hạng 1

- `class_id` = 468
- `class_name` = cab
- `rank` = 1
- `score` = 0.510915
- `taxonomy_name` = ImageNet-1K

### 1.2. Record này nói về cả bức ảnh hay về một vật thể riêng? Vì sao?

Record này nói về **toàn bộ bức ảnh**, không phải một vật thể riêng.

Vì đây là bài toán Image Classification, mô hình dự đoán nhãn cho toàn bộ ảnh mà không xác định vị trí hoặc bounding box của từng vật thể.

Kết quả Top-1 là `cab` với score 0.510915, tức đây là lớp được mô hình xếp hạng cao nhất.

Tuy nhiên, ảnh `traffic` có nhiều phương tiện như xe buýt, ô tô và các phương tiện khác. Vì vậy, không thể hiểu rằng tất cả phương tiện trong ảnh đều thuộc lớp `cab`.

### 1.3. Danh sách tên lớp (class list) do đâu mà có?

Danh sách lớp được định nghĩa sẵn theo taxonomy mà checkpoint sử dụng, không phải do mô hình tự tạo ra khi dự đoán.

Cụ thể, `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K gồm 1.000 lớp. Vì vậy, mô hình chỉ có thể đưa ra dự đoán thuộc các lớp trong danh sách này.

### 1.4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`?

- `class_id`: Mã số định danh lớp, thuận tiện cho xử lý bằng máy.
- `class_name`: Tên lớp giúp con người đọc và hiểu kết quả.
- `taxonomy_name`: Xác định hệ thống phân loại đang sử dụng, ví dụ ImageNet-1K hoặc COCO-80.

Cùng một `class_id` có thể tương ứng với những lớp khác nhau trong các taxonomy khác nhau.

Ghi đủ ba trường giúp tránh nhầm lẫn, đảm bảo tính nhất quán và hỗ trợ kiểm tra dữ liệu.

### 1.5. Nếu ảnh có nhiều vật/nhiều chủ đề, guideline cần nói rõ điều gì?

Guideline cần quy định rõ tiêu chí chọn một nhãn đại diện khi ảnh chứa nhiều vật thể hoặc nhiều chủ đề.

Ví dụ:

- Xác định vật thể chính hoặc nội dung nổi bật nhất của ảnh.
- Quy định có ưu tiên vật thể chiếm diện tích lớn, ở trung tâm hay có vai trò chính trong ngữ cảnh ảnh hay không.
- Chỉ lựa chọn nhãn thuộc taxonomy đang sử dụng.
- Nếu không thể xác định được nhãn phù hợp, cần chuyển lên Lab Coach để thống nhất cách xử lý.

Những tiêu chí này phải được quy định trước để các annotator gán nhãn nhất quán.

### 1.6. Vì sao không được lấy `score` của model làm ground truth?

Model score chỉ thể hiện mức độ tin cậy của mô hình đối với một dự đoán, không chứng minh dự đoán đó chính xác.

Mô hình vẫn có thể dự đoán sai dù score cao hoặc bị ảnh hưởng bởi dữ liệu huấn luyện.

Ví dụ, ảnh `traffic` chứa nhiều phương tiện nhưng mô hình dự đoán lớp `cab` với score 0.510915. Đây chỉ là kết quả dự đoán, chưa đủ căn cứ để coi `cab` là ground truth.

Ground truth phải được xác lập theo guideline và kiểm tra chất lượng, không được tự động lấy nhãn có score cao nhất làm nhãn đúng.

---

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.

Chỉ sử dụng ảnh `sample_id = kitchen`.

### 2.1. Chép một record

- `class_name` = person
- `score` = 0.912625
- `bbox_xyxy` = [385.33, 69.24, 498.92, 348.92]
- `bbox_width` = 113.58
- `bbox_height` = 279.68

### 2.2. Giải thích box bằng lời thường

Bounding box là hình chữ nhật mô hình dự đoán bao quanh người đầu bếp trong ảnh `kitchen`.

- Góc trên-trái: (385.33, 69.24) pixel.
- Góc dưới-phải: (498.92, 348.92) pixel.
- Chiều rộng ghi trong JSON: 113.58 pixel.
- Chiều cao: 279.68 pixel.

Hệ tọa độ lấy gốc (0, 0) tại góc trên-trái của ảnh.

Bounding box xác định vùng chứa vật thể, nhưng không thể mô tả chính xác đường viền cơ thể như segmentation.

### 2.3. Chọn hai ngưỡng (ví dụ 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

Kết quả chạy mô hình trên ảnh `kitchen`:

| Threshold | Số prediction |
|---|---:|
| 0.20 | 17 |
| 0.35 | 11 |
| 0.60 | 6 |

Khi tăng threshold từ 0.20 lên 0.60, số prediction được giữ lại giảm từ 17 xuống 6.

Một số prediction có score thấp như `cup` 0.45 và `bowl` 0.46 được giữ ở threshold 0.20 nhưng bị loại khi tăng threshold lên 0.60.

Điều này cho thấy threshold ảnh hưởng đến số prediction được giữ lại, không quyết định vật thể nào thực sự tồn tại trong ground truth.

### 2.4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?

Khi hạ threshold, mô hình giữ lại nhiều prediction có score thấp hơn. Vì vậy, reviewer cần kiểm tra thêm để:

- Phát hiện các prediction sai nhãn (false positive).
- Kiểm tra bounding box không chính xác hoặc bị trùng trên cùng một vật thể.
- Phân biệt các vật thể riêng biệt có bounding box chồng lấn do che khuất.
- Kiểm tra các vật thể thật bị bỏ sót (false negative).

Ví dụ, tại threshold 0.20 có 17 prediction, trong khi threshold 0.60 chỉ có 6 prediction.

Do đó, khi sử dụng threshold thấp, reviewer phải rà soát nhiều prediction hơn và không được tự động coi prediction có score thấp là sai.

### 2.5. Viết một quy tắc box để người gán nhãn làm theo

> Vẽ bounding box bao sát phần nhìn thấy được của từng vật thể, hạn chế khoảng trống chứa nền và không cắt vào vật thể. Nếu vật thể bị che khuất, chỉ bao quanh phần nhìn thấy được, không tự suy đoán phần bị che. Nếu vật thể bị cắt bởi mép ảnh, bounding box phải dừng tại mép ảnh. Các vật thể riêng biệt vẫn cần box riêng, kể cả khi các box chồng lấn do che khuất.

Quy tắc này giúp annotator tạo bounding box nhất quán, đồng thời tránh đưa vào nhãn những phần vật thể không quan sát được.

### 2.6. Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Trong ảnh `kitchen`, chiếc bếp `oven` bên trái (score 0.686791) bị cắt ở mép trái ảnh và một phần phía dưới bị bàn che khuất.

Em chọn **đưa vào guideline** vì vật thể vẫn có thể nhận diện được và đây là trường hợp thường gặp.

**Quy tắc:** Với vật thể bị che khuất hoặc cắt mép nhưng vẫn xác định được nhãn, 
vẽ bounding box bao sát phần nhìn thấy được, không vượt ra ngoài ảnh và không tự suy đoán phần bị che. 
Nếu không xác định được vật thể thì chuyển lên Lab Coach.

---

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.

Chỉ sử dụng ảnh `sample_id = kitchen`.

### 3.1. Chép một record

- `instance_id` = kitchen-001
- `class_name` = person
- `score` = 0.899318
- `polygon_point_count` = 348
- 5 điểm đầu của `polygon_xy` = [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

### 3.2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

Polygon mô tả đường viền của vật thể chi tiết hơn so với bounding box.

Bounding box chỉ là hình chữ nhật bao quanh vật thể nên thường chứa cả những vùng nền bên trong hình chữ nhật đó.

Trong khi đó, polygon được tạo từ nhiều điểm tọa độ, giúp mô tả hình dạng và vùng chiếm chỗ của vật thể sát với đường biên hơn.

Ví dụ, đối với người đầu bếp trong ảnh `kitchen`, polygon có thể mô tả đường viền đầu, vai, tay, thân và chân thay vì chỉ tạo một hình chữ nhật.

Tuy nhiên, polygon vẫn là prediction của mô hình nên có thể chứa sai sót, không mặc định là đường viền ground truth chính xác.

### 3.3. `instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?

`instance_id` là mã định danh riêng cho từng vật thể được mô hình phân đoạn trong ảnh.

- **Class ID:** Xác định vật thể thuộc lớp nào. Ví dụ, các bát đều thuộc lớp `bowl`.
- **Instance ID:** Phân biệt từng bát riêng biệt, dù chúng cùng thuộc một lớp.
- **Tracking ID:** Dùng trong bài toán theo dõi vật thể qua nhiều frame video, giúp nhận biết cùng một vật thể ở các thời điểm khác nhau.

Ví dụ, trong ảnh `kitchen`, nhiều bát có thể cùng mang class `bowl`, nhưng mỗi bát được phân đoạn riêng sẽ có một `instance_id` khác nhau.

Trong bài lab này, `instance_id` chỉ là mã do notebook tạo để phân biệt các prediction, không phải ID theo dõi xuyên suốt các frame video.

### 3.4. Viết một quy tắc vẽ biên mask để người gán nhãn làm theo

> Vẽ polygon bám sát đường viền phần nhìn thấy được của vật thể, hạn chế đưa pixel nền vào mask. Mỗi vật thể riêng biệt phải có polygon và instance ID riêng, kể cả khi chúng thuộc cùng một lớp. Khi vật thể bị che khuất, chỉ gán nhãn phần có thể quan sát được theo guideline, không tự suy đoán ranh giới bị che. Với những vùng biên quá mờ hoặc không thể phân biệt rõ, cần chuyển reviewer hoặc Lab Coach để thống nhất cách xử lý.

Không đặt giới hạn cứng về số điểm polygon nếu guideline chưa quy định, vì vật thể có hình dạng phức tạp có thể cần nhiều điểm để mô tả đường viền.

### 3.5. Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Trong ảnh `kitchen`, các bát (`bowl`) ở góc dưới bên trái bàn nằm sát nhau, một phần bị che khuất khiến ranh giới giữa các vật thể khó phân biệt. Đặc biệt, các vùng dự đoán gần bát có score 0.74 xuất hiện chồng lấn.

Em chọn **escalation (hỏi Lab Coach)** vì chưa thể xác định chắc chắn đường biên của từng vật thể chỉ từ kết quả segmentation. Nếu tự vẽ theo suy đoán, các annotator có thể tạo mask không nhất quán.

Sau khi được hướng dẫn, mỗi bát sẽ được gán một mask và `instance_id` riêng theo guideline.
---

## 4. Vòng đời và kiểm tra chất lượng

Bài lab hôm nay chỉ sử dụng mô hình đã huấn luyện để quan sát và phân tích prediction, không thực hiện huấn luyện hoặc tinh chỉnh.

Quy trình tổng quát:

`Ảnh thô → Guideline → Ground truth → Huấn luyện → Prediction → QC / Rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| Phân loại ảnh | Một nhãn cho toàn bộ ảnh theo taxonomy và guideline. | Ảnh `traffic` có nhiều phương tiện như bus, car nhưng Top-1 là `cab` (0.510915). Khó chọn một nhãn đại diện cho toàn cảnh. | Chọn nhãn phù hợp cho toàn ảnh theo taxonomy và guideline, không sao chép prediction thành ground truth. | Kiểm tra nhãn có phù hợp nội dung, đúng taxonomy và nhất quán giữa các ảnh không. |
| Phát hiện vật thể | Danh sách bounding box và class cho từng vật thể. | Ở threshold 0.20 có 17 prediction, còn 0.60 chỉ có 6. Một số vật thể nhỏ có score thấp và một người ở mép trái bị cắt khung hình. | Vẽ tight box cho từng vật thể, kể cả vật thể bị che hoặc cắt mép nếu đủ điều kiện theo guideline. | Kiểm tra box sai, box trùng trên cùng vật thể, vật thể bỏ sót và nhãn sai. |
| Instance Segmentation | Polygon hoặc mask của từng vật thể, kèm class và instance ID riêng. | Khu vực nhiều bowl trên bàn có các prediction gần nhau, khó phân biệt chính xác ranh giới một số vật thể. | Vẽ polygon sát phần vật thể nhìn thấy được, phân biệt các instance riêng biệt. | Kiểm tra đường biên mask, nhãn, vật thể bị bỏ sót và tính nhất quán khi xử lý vùng bị che khuất. |

---

## 5. An toàn dữ liệu

### 5.1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì?

Chỉ sử dụng ba ảnh COCO 2017 validation công khai được notebook cung cấp, 
không tải ảnh cá nhân hoặc dữ liệu nội bộ lên Colab/GitHub và 
phải giữ đầy đủ thông tin ghi công tác giả, nguồn, giấy phép khi sử dụng ảnh.

### 5.2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

Nếu gặp ảnh cá nhân, dữ liệu nội bộ hoặc dữ liệu không đúng phạm vi bài lab, em sẽ dừng xử lý và báo ngay cho Lab Coach hoặc giảng viên phụ trách để được hướng dẫn.

Em không tự ý chia sẻ, tiếp tục sử dụng hoặc xóa dữ liệu khi chưa có hướng dẫn xử lý phù hợp.

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền).
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
- [x] Đã nộp **link repository** trên LMS.

*Chỉ đánh dấu `[x]` sau khi đã kiểm tra hoặc thực hiện thành công từng mục.*

