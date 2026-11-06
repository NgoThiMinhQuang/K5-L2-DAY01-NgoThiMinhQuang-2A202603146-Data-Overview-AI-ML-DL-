# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Python 3.11 / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

---

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số "tin" của model; không phải điểm chất lượng nhãn người |

---

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

**1. Record hạng 1:**
- `class_id` = 468
- `class_name` = cab
- `rank` = 1
- `score` = 0.510915
- `taxonomy_name` = ImageNet-1K

**2. Record này nói về cả bức ảnh hay về một vật thể riêng?**
Record này nói về **cả bức ảnh**, không phải một vật thể riêng. 
Lý do: đây là task **image classification** (phân loại ảnh), 
model chỉ đưa ra **một nhãn duy nhất** cho toàn bộ bức ảnh, 
không xác định vị trí hay bounding box của vật thể nào. 
Score 0.510915 là độ tin cậy rằng "toàn bộ bức ảnh này thuộc lớp cab".

**3. Danh sách tên lớp (class list) do đâu mà có?**
Danh sách lớp **đi kèm taxonomy/checkpoint**, không phải model tự nghĩ ra. 
Cụ thể, `yolo11n-cls.pt` được huấn luyện trên bộ **ImageNet-1K** (1000 lớp cố định), 
nên model chỉ có thể dự đoán trong 1000 lớp này.

**4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`?**
- `class_id`: định danh số, tránh trùng lặp tên, dễ xử lý bằng máy.
- `class_name`: dễ đọc, dễ hiểu với con người.
- `taxonomy_name`: cho biết **bộ từ điển nào** đang được dùng (ImageNet-1K vs COCO-80). 
Cùng một `class_id=10` nhưng ở hai taxonomy khác nhau sẽ là hai lớp hoàn toàn khác nhau. 
Ghi cả 3 trường giúp **reproducible** và tránh nhầm lẫn khi trao đổi dữ liệu.

**5. Nếu ảnh có nhiều vật/nhiều chủ đề, guideline cần nói rõ điều gì?**
Guideline cần quy định rõ **tiêu chí ưu tiên** khi chọn một lớp duy nhất, ví dụ:
- Chọn lớp của **vật thể chiếm diện tích lớn nhất** trong ảnh.
- Hoặc chọn lớp của **vật thể ở trung tâm** ảnh.
- Hoặc chọn **chủ đề tổng quát nhất** (vd. "street scene" thay vì "car").
- Nếu không rõ → **escalation** lên Lab Coach, không tự quyết định.

**6. Vì sao không được lấy `score` của model làm ground truth?**
Vì **model score chỉ phản ánh độ tin cậy của model**, không phải chất lượng nhãn thật. Model có thể:
- Dự đoán sai với score cao (false positive tự tin).
- Bỏ sót vật thể thật (false negative).
- Bị ảnh hưởng bởi bias trong dữ liệu huấn luyện.
Ground truth phải do **con người gán theo guideline**, không phải do model sinh ra.

---

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen`.

**1. Chép một record:**
- `class_name` = person
- `score` = 0.912625
- `bbox_xyxy` = [385.33, 69.24, 498.92, 348.92]
- `bbox_width` = 113.58
- `bbox_height` = 279.68

**2. Giải thích box bằng lời thường:**
Box này là hình chữ nhật bao quanh người đầu bếp (person) trong ảnh kitchen. 
Góc trên-trái của box nằm ở tọa độ (385.33, 69.24) pixel, tính từ gốc tọa độ (0, 0) ở góc trên-trái của ảnh. 
Góc dưới-phải nằm ở tọa độ (498.92, 348.92) pixel. 
Box có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel, bao kín toàn bộ phần cơ thể người đầu bếp từ đầu đến chân.

**3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**
- Ngưỡng **0.20**: Có **17** prediction (vật thể).
- Ngưỡng **0.60**: Có **6** prediction (vật thể).
*(Khi hạ ngưỡng xuống 0.20, model giữ lại nhiều box hơn nhưng đi kèm các vật thể có độ tin cậy thấp như cup 0.45, bowl 0.46).*

**4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?**
Khi hạ ngưỡng, số lượng box tăng lên đáng kể (từ 6 lên 17), 
trong đó xuất hiện nhiều box có score thấp. 
Người reviewer phải tốn thêm công sức để **kiểm tra và lọc bỏ các false positive** (các box nhận diện nhầm) và
 **xử lý các box trùng lặp** (duplicate boxes) vẽ chồng lên cùng một vật thể, 
để đảm bảo ground truth chỉ giữ lại các nhãn chính xác.

**5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).**
> "Vẽ bounding box dạng tight box (hộp chặt), bao kín toàn bộ phần nhìn thấy được của vật thể, 
không chừa khoảng trống thừa và không cắt vào vật thể. 
Nếu vật thể bị che khuất bởi vật khác, chỉ vẽ box cho phần lộ ra. 
Nếu vật thể bị cắt mép ở biên ảnh, vẽ box sát mép ảnh."

**6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**

Trên ảnh `kitchen`, vật thể **`oven` với score 0.63** (box màu tím ở phía bên phải ảnh) là một vật bị **cắt mép ở biên ảnh**. 
Chỉ thấy một phần nhỏ của lò nướng này ở cạnh phải, phần còn lại bị cắt ra khỏi khung hình.

Em sẽ đưa trường hợp này lên **escalation (hỏi Lab Coach)**.
**Vì sao:**
1. **Không rõ đây có phải vật thể cần gán nhãn không:** Chỉ thấy một phần nhỏ của oven, 
không thể xác định chắc chắn đây là lò nướng thật hay model nhận diện nhầm dựa trên vài pixel ở biên ảnh.
2. **Guideline chưa rõ về vật thể cắt mép:** Không có quy định cụ thể "có gán nhãn cho vật thể bị cắt mép ở biên ảnh không?" 
và "nếu có, vẽ box đến sát mép ảnh hay bỏ qua?".
3. **Score thấp (0.63):** Độ tin cậy không cao, có thể là false positive do model chỉ thấy một phần rất nhỏ của vật thể.
4. **Nguy cơ inconsistent:** Annotator A có thể vẽ box nhỏ cho phần nhìn thấy, Annotator B có thể bỏ qua hoàn toàn.
 Việc tự ý quyết định sẽ gây **nhãn nhiễu (noisy labels)** trong ground truth.
5. **Nguyên tắc của lab:** "Prediction không phải ground truth" - không được tự suy đoán phần bị cắt mép. 
Hỏi Lab Coach đảm bảo tính đồng bộ cho dataset.

--- 
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`.

**1. Chép một record:**
- `instance_id` = kitchen-001
- `class_name` = person
- `score` = 0.899318
- số điểm polygon = `polygon_point_count` = 348
- 3–5 điểm đầu của `polygon_xy` = [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

**2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**
Polygon mô tả **hình dáng thực tế** của vật thể thay vì chỉ là hình chữ nhật bao quanh. 
Ví dụ với vật thể "person" (người đầu bếp), box chỉ là hình chữ nhật lớn bao quanh toàn bộ cơ thể (kể cả khoảng trống hai bên tay, chân), 
trong khi polygon bám sát từng đường viền cơ thể: vai, tay, eo, chân. 
Điều này giúp loại bỏ các pixel nền không thuộc vật thể, 
cung cấp thông tin chính xác hơn về hình dạng, 
kích thước thật và vị trí chi tiết của vật thể trong ảnh.

**3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào? **
`instance_id` (ví dụ: `kitchen-002`) là **định danh duy nhất cho từng vật thể cụ thể (instance)** trong một bức ảnh, giúp phân biệt các vật thể cùng lớp. 
- **Không phải class ID**: class ID chỉ phân loại vật thể thuộc lớp nào (vd: class_id=0 là "person"), 
nhưng trong ảnh kitchen có thể có 2 người → cần 2 instance_id khác nhau (kitchen-002 và kitchen-005) để phân biệt từng người riêng biệt.
- **Không phải tracking ID**: tracking ID dùng để theo dõi cùng một vật thể qua nhiều frame trong video 
(vd: người này di chuyển từ frame 1 sang frame 2 vẫn giữ nguyên ID). 
Ở đây chỉ là ảnh tĩnh nên instance_id chỉ có ý nghĩa trong phạm vi một bức ảnh duy nhất.


**4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.**
> "Vẽ polygon bám sát viền vật thể với sai số không quá 2-3 pixel. 
Tại vùng biên mờ (shadow, reflection, chuyển màu), vẽ theo ranh giới mà mắt người có thể phân biệt được giữa vật thể và nền. 
Nếu hai vật thể cùng lớp chạm nhau, phải vẽ đường phân chia rõ ràng giữa chúng thành 2 polygon riêng biệt, 
mỗi polygon có instance_id riêng. Số điểm polygon tối thiểu là 3, không vẽ quá 50 điểm để tránh overfitting. 
Polygon phải nằm hoàn toàn bên trong bounding box của vật thể."

**5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**
Trên ảnh `kitchen`, có **hai cái bowl (bát)** với instance_id `kitchen-003` (score 0.698698) và `kitchen-006` (score 0.528201) đặt sát nhau trên bàn. Hai bounding box của chúng chồng lấn nhau:
- `kitchen-003`: bbox [57.21, 288.5, 132.99, 327.92]
- `kitchen-006`: bbox [0.11, 302.85, 85.7, 370.15]
Viền giữa hai bowl rất mờ và khó phân biệt rõ ràng trong ảnh. Đây là trường hợp cần **escalation (hỏi Lab Coach)**.
**Lý do:**
- Guideline hiện tại chưa rõ: khi hai vật thể cùng lớp (cùng là "bowl") có bbox chồng lấn và viền mờ, annotator nên vẽ 2 polygon riêng biệt hay gộp thành 1 polygon chung?
- Nếu vẽ riêng, đường phân chia nên đặt ở đâu khi mắt người cũng khó xác định ranh giới chính xác?
- Việc tự ý quyết định sẽ gây **inconsistent** (không nhất quán) giữa các annotator, ảnh hưởng trực tiếp đến chất lượng training data cho model segmentation.
---
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất (class_name) cho toàn bộ ảnh, kèm taxonomy_name (vd: ImageNet-1K). | Ảnh `traffic` thực tế là cảnh đường phố với nhiều xe bus, car, nhưng model dự đoán top-1 là `cab` (score 0.51) do dùng taxonomy ImageNet-1K khác với COCO. | Chọn **một** lớp duy nhất cho cả ảnh theo đúng guideline và taxonomy được giao, không dựa vào prediction của model. | Kiểm tra class_name có khớp với taxonomy không, có nhất quán với các ảnh cùng loại không, và không bị ảnh hưởng bởi model score. |
| Phát hiện vật thể | Danh sách các bbox_xyxy (4 tọa độ pixel) + class_name cho mỗi vật thể trong ảnh. | Ở threshold 0.60 chỉ có 6 vật thể (bỏ sót nhiều bowl, cup), ở threshold 0.20 có 17 vật thể (nhiều box điểm thấp như bowl 0.46, cup 0.45 có thể là false positive). Có 2 box `oven` (0.69 và 0.63) chồng lên nhau. | Vẽ tight box bao kín từng vật thể nhìn thấy được, mỗi vật thể một box riêng, không vẽ chồng lên nhau. | So sánh box với ảnh gốc: box có tight không, có bỏ sót vật thể nào không, có box trùng lặp hay false positive không, class_name có đúng không. |
| Instance segmentation | Danh sách các polygon_xy (nhiều điểm [x,y]) + instance_id duy nhất + class_name cho mỗi vật thể. | Nhiều `bowl` (0.70 và 0.53) đặt sát nhau trên bàn, viền giữa chúng mờ và khó tách. Model có thể gộp hoặc tách sai các vật thể cùng lớp chạm nhau. | Vẽ polygon bám sát viền từng vật thể, gán instance_id riêng (vd: kitchen-001, kitchen-002) cho mỗi bowl dù cùng lớp. | Kiểm tra polygon có bám sát viền không, instance_id có duy nhất không, các vật thể chạm nhau đã được tách đúng thành polygon riêng chưa. |

--- 
## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì?**
   > "Chỉ sử dụng dữ liệu ảnh công khai từ COCO 2017 validation set có giấy phép CC BY 2.0, 
tuyệt đối không tải lên, chia sẻ hoặc sử dụng dữ liệu cá nhân, 
dữ liệu nội bộ nhạy cảm của tổ chức trong quá trình làm bài."

2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**
   > "Dừng ngay việc xử lý và gán nhãn, sau đó báo cáo cho **Lab Coach / giảng viên phụ trách** để được hướng dẫn. 
Không tự ý xóa, chia sẻ, hoặc tiếp tục sử dụng dữ liệu đó."

---

## 6. Checklist trước khi nộp link repo

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS