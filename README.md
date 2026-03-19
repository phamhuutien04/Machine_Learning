# HỆ THỐNG NHẬN DẠNG BỆNH LÁ CÂY

Hệ thống Machine Learning sử dụng **MobileNetV2** để trích xuất đặc trưng và **KNN/SVM** để phân loại 38 loại bệnh lá cây khác nhau.

---

## 📋 MỤC LỤC

- [Tổng quan](#-tổng-quan)
- [Cách hoạt động](#-cách-hoạt-động-của-mô-hình)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Dataset](#-dataset)
- [Kết quả](#-kết-quả)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Giải thích chi tiết](#-giải-thích-chi-tiết)

---

## 🎯 TỔNG QUAN

### Mục tiêu
Xây dựng hệ thống tự động nhận dạng bệnh trên lá cây từ ảnh, giúp nông dân phát hiện sớm và xử lý kịp thời.

### Đặc điểm
- **38 loại bệnh** trên nhiều loại cây (táo, nho, cà chua, khoai tây...)
- **54,305 ảnh** (43,444 train + 10,861 validation)
- **Độ chính xác**: KNN ~92%, SVM ~93%
- **Không cần GPU** để inference (chỉ cần khi train)

---

## 🔄 CÁCH HOẠT ĐỘNG CỦA MÔ HÌNH

### Pipeline tổng quan

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌──────────┐
│  Ảnh lá cây │ ───> │ MobileNetV2  │ ───> │ StandardScaler│ ───>│ KNN/SVM  │
│ (224x224x3) │      │   (Feature   │      │  (Chuẩn hóa)  │     │(Phân loại)│
└─────────────┘      │  Extractor)  │      └─────────────┘      └──────────┘
                     │              │
                     │ Vector 1280  │
                     │    chiều     │
                     └──────────────┘
```

### Các bước chi tiết

#### **Bước 1: Tiền xử lý ảnh**
```python
# Input: Ảnh RGB kích thước bất kỳ
# Output: Ảnh 224x224x3, giá trị pixel [0, 1]

1. Resize ảnh về 224x224 pixels
2. Chuẩn hóa giá trị pixel: chia cho 255
   - Từ [0, 255] → [0, 1]
```

**Tại sao 224x224?**
- Kích thước chuẩn của MobileNetV2
- Cân bằng giữa chi tiết và tốc độ xử lý

#### **Bước 2: Trích xuất đặc trưng (MobileNetV2)**
```python
# Input: Ảnh 224x224x3
# Output: Vector 1280 số thực

MobileNetV2 (pretrained trên ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Vector [f1, f2, f3, ..., f1280]
```

**MobileNetV2 làm gì?**
- Mạng neural network đã được train trên 1.4 triệu ảnh
- Trích xuất các đặc trưng: màu sắc, hình dạng, texture, pattern
- Mỗi ảnh → 1 vector 1280 số

**Ví dụ:**
```
Ảnh lá bệnh đốm đen
    ↓
[0.23, -0.45, 0.89, 0.12, ..., -0.67]
 ↑      ↑      ↑     ↑          ↑
màu   hình   vết   texture   pattern
sắc   dạng   bệnh
```

#### **Bước 3: Chuẩn hóa (StandardScaler)**
```python
# Input: Vector 1280 chiều
# Output: Vector 1280 chiều đã chuẩn hóa

Công thức: x_scaled = (x - mean) / std

Ví dụ:
Trước: [100, 200, 300, 50, 150]
Mean = 160, Std = 94.87
Sau:  [-0.63, 0.42, 1.48, -1.16, -0.11]
```

**Tại sao cần chuẩn hóa?**
- KNN và SVM nhạy cảm với scale của dữ liệu
- Đảm bảo tất cả features có tầm quan trọng ngang nhau
- Tăng độ chính xác 5-10%

#### **Bước 4: Phân loại**

##### **4A. K-Nearest Neighbors (KNN)**
```
Nguyên lý: "Cho tôi biết bạn là ai, tôi sẽ nói bạn là người như thế nào"

1. Có 1 ảnh lá mới cần phân loại
2. Tính khoảng cách đến TẤT CẢ ảnh trong tập train
3. Chọn K=5 ảnh gần nhất
4. Đếm xem 5 ảnh đó thuộc loại bệnh nào nhiều nhất
5. Kết luận: Ảnh mới thuộc loại đó
```

**Công thức khoảng cách Euclidean:**
```
distance = √[(x₁-y₁)² + (x₂-y₂)² + ... + (x₁₂₈₀-y₁₂₈₀)²]
```

**Ví dụ cụ thể:**
```
Ảnh mới: Lá táo có đốm đen
         ↓
Tìm 5 ảnh gần nhất:
  1. Apple Scab (khoảng cách: 0.12)
  2. Apple Scab (khoảng cách: 0.15)
  3. Apple Scab (khoảng cách: 0.18)
  4. Black Rot  (khoảng cách: 0.20)
  5. Apple Scab (khoảng cách: 0.21)
         ↓
Đếm: Apple Scab = 4, Black Rot = 1
         ↓
Kết luận: Apple Scab (80% confidence)
```

##### **4B. Support Vector Machine (SVM)**
```
Nguyên lý: "Tìm đường thẳng/mặt phẳng tốt nhất để phân chia các nhóm"

1. Tìm siêu phẳng (hyperplane) phân chia các class
2. Chọn siêu phẳng có margin (khoảng cách) lớn nhất
3. Khi có ảnh mới: xem nó nằm bên nào của siêu phẳng
```

**Minh họa 2D (đơn giản hóa):**
```
        Apple Scab
    ●     ●     ●
  ●   ●     ●
─────────────────── ← Siêu phẳng (đường phân chia)
      ○   ○   ○
    ○       ○     ○
        Black Rot
```

**Công thức:**
```
f(x) = w·x + b

Nếu f(x) > 0 → Class A
Nếu f(x) < 0 → Class B
```

**Multi-class (38 classes):**
- Tạo 38 bộ phân loại One-vs-Rest
- Mỗi bộ: "Class này" vs "Tất cả class khác"
- Chọn class có confidence cao nhất

---

## 🏗️ KIẾN TRÚC HỆ THỐNG

### 1. Feature Extractor: MobileNetV2

```
Input: 224x224x3
    ↓
Conv2D (32 filters)
    ↓
Inverted Residual Blocks (×17)
    ↓
Conv2D (1280 filters)
    ↓
GlobalAveragePooling2D
    ↓
Output: 1280 features
```

**Đặc điểm:**
- Pretrained trên ImageNet
- Nhẹ, nhanh (thiết kế cho mobile)
- Không cần train lại (transfer learning)

### 2. Classifiers

#### KNN (K=5)
```python
KNeighborsClassifier(n_neighbors=5)
```
- Không cần training
- Lưu toàn bộ training data
- Prediction: O(n) - chậm với dataset lớn

#### SVM (LinearSVC)
```python
LinearSVC(max_iter=3000)
```
- Cần training
- Chỉ lưu weights (w, b)
- Prediction: O(1) - rất nhanh

---

## � DATASET

### PlantVillage Dataset

**Tổng quan:**
- 54,305 ảnh lá cây
- 38 classes (14 loại cây × nhiều bệnh)
- Ảnh chụp trong điều kiện kiểm soát

**Phân chia:**
- Training: 43,444 ảnh (80%)
- Validation: 10,861 ảnh (20%)

**Các loại cây:**
- Apple (Táo)
- Blueberry (Việt quất)
- Cherry (Anh đào)
- Corn (Ngô)
- Grape (Nho)
- Orange (Cam)
- Peach (Đào)
- Pepper (Ớt)
- Potato (Khoai tây)
- Raspberry (Mâm xôi)
- Soybean (Đậu nành)
- Squash (Bí)
- Strawberry (Dâu tây)
- Tomato (Cà chua)

**Các loại bệnh:**
- Scab (Ghẻ)
- Black rot (Thối đen)
- Cedar apple rust (Gỉ sắt)
- Powdery mildew (Phấn trắng)
- Bacterial spot (Đốm vi khuẩn)
- Early blight (Héo sớm)
- Late blight (Héo muộn)
- Leaf Mold (Nấm lá)
- Septoria leaf spot (Đốm lá)
- Spider mites (Nhện đỏ)
- Target Spot (Đốm mục tiêu)
- Yellow Leaf Curl Virus (Virus cuộn lá vàng)
- Mosaic virus (Virus khảm)
- Healthy (Khỏe mạnh)

---

## 📈 KẾT QUẢ

### Độ chính xác

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| KNN   | 91.89%   | 91%       | 89%    | 90%      |
| SVM   | 93.36%   | 91%       | 92%    | 91%      |

### Thời gian xử lý

**Lần đầu (train từ đầu):**
```
1. Feature Extraction: ~16 phút (90% thời gian)
2. KNN Training:       ~33 giây (0.4% thời gian)
3. SVM Training:       ~110 phút (86.9% thời gian)
───────────────────────────────────────────────
TỔNG:                  ~126 phút
```

**Lần sau (load models):**
```
1. Feature Extraction: ~16 phút
2. KNN Prediction:     ~33 giây
3. SVM Prediction:     ~1 giây
───────────────────────────────────────────────
TỔNG:                  ~17 phút
```

**Inference (1 ảnh mới):**
```
1. Preprocess:         ~0.01s
2. Feature Extract:    ~0.05s
3. Prediction:         ~0.001s (SVM) / ~0.1s (KNN)
───────────────────────────────────────────────
TỔNG:                  ~0.06s (SVM) / ~0.16s (KNN)
```

### Top 10 Classes (Accuracy cao nhất)

1. Raspberry healthy: 100%
2. Grape Leaf blight: 100%
3. Peach healthy: 100%
4. Orange Haunglongbing: 99%
5. Squash Powdery mildew: 99%
6. Strawberry healthy: 99%
7. Corn healthy: 99%
8. Soybean healthy: 99%
9. Cherry Powdery mildew: 98%
10. Tomato Yellow Leaf Curl Virus: 98%

---

## 💻 CÀI ĐẶT

### Yêu cầu hệ thống
- Python 3.8+
- RAM: 8GB+ (16GB khuyến nghị)
- Disk: 5GB+ (cho dataset và models)
- GPU: Không bắt buộc (chỉ tăng tốc training)

### Cài đặt thư viện

```bash
pip install tensorflow
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install numpy
pip install joblib
```

Hoặc:
```bash
pip install -r requirements.txt
```

### Tải dataset

```bash
# Download PlantVillage dataset
# Link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

# Giải nén vào thư mục archive/
unzip plant-village.zip -d archive/
```

Cấu trúc thư mục:
```
archive/
├── PlantVillage/
│   ├── train/
│   │   ├── Apple___Apple_scab/
│   │   ├── Apple___Black_rot/
│   │   └── ...
│   └── val/
│       ├── Apple___Apple_scab/
│       ├── Apple___Black_rot/
│       └── ...
```

---

## 🚀 SỬ DỤNG

### 1. Training (lần đầu)

```bash
# Chạy Jupyter Notebook
jupyter notebook plant_disease_analysis.ipynb



**Lưu ý:** Training SVM mất ~2 giờ. Hãy kiên nhẫn!

### 2. Load models đã train

```python
from joblib import load

# Load models
scaler = load('models/scaler.joblib')
knn = load('models/knn_model.joblib')
svm = load('models/svm_model.joblib')

# Load class names
import json
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)
```

### 3. Predict ảnh mới

```python
import tensorflow as tf
import numpy as np

# Load ảnh
img = tf.keras.preprocessing.image.load_img(
    'test_image.jpg', 
    target_size=(224, 224)
)
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Extract features
extractor = tf.keras.models.load_model('models/extractor.h5')
features = extractor.predict(img_array)

# Scale
features_scaled = scaler.transform(features)

# Predict
prediction = svm.predict(features_scaled)
disease = class_names[prediction[0]]

print(f"Bệnh: {disease}")
```

---

## 📚 GIẢI THÍCH CHI TIẾT

### Tại sao chạy lâu?

**Phân tích thời gian:**
```
Feature Extraction: 90% thời gian
    - MobileNetV2 xử lý 54,305 ảnh
    - Mỗi ảnh qua 17 layers
    - CPU: ~16 phút
    - GPU: ~3 phút

Training: 10% thời gian
    - KNN: Chỉ lưu data (~33s)
    - SVM: Tìm siêu phẳng tối ưu (~110 phút)
```

**Tối ưu hóa:**
1. Lưu features sau khi extract:
```python
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
# Lần sau load luôn → tiết kiệm 16 phút!
```

2. Dùng GPU cho MobileNetV2:
```python
# Tăng tốc 5-10 lần
with tf.device('/GPU:0'):
    features = extractor.predict(images)
```

3. Giảm max_iter của SVM:
```python
# Nếu đã hội tụ sớm
svm = LinearSVC(max_iter=1000)
```

### So sánh KNN vs SVM

| Tiêu chí | KNN | SVM |
|----------|-----|-----|
| **Training** | Không cần (chỉ lưu data) | Cần (~110 phút) |
| **Prediction** | Chậm (tính khoảng cách) | Nhanh (w·x+b) |
| **Bộ nhớ** | Lớn (lưu 43,444 vectors) | Nhỏ (chỉ w, b) |
| **Accuracy** | 91.89% | 93.36% |
| **Giải thích** | Dễ (xem K láng giềng) | Khó |
| **Phù hợp** | Research, prototyping | Production |

### Confusion Matrix

**Cách đọc:**
```
                Predicted
              A    B    C
        A   [50    2    1]  ← 50 đúng, 3 sai
Actual  B   [ 1   48    2]  ← 48 đúng, 3 sai
        C   [ 0    1   49]  ← 49 đúng, 1 sai
```

- **Đường chéo** (màu đậm): Dự đoán đúng
- **Ngoài đường chéo**: Dự đoán sai
- Ô [A, B] = 2: Có 2 ảnh thực tế là A nhưng dự đoán là B

### Metrics

**1. Accuracy (Độ chính xác)**
```
Accuracy = Số dự đoán đúng / Tổng số dự đoán
         = 10,000 / 10,861
         = 92.07%
```

**2. Precision (Độ chính xác dương)**
```
Precision = TP / (TP + FP)

Ví dụ:
- Dự đoán 100 ảnh là Apple Scab
- Thực tế chỉ 90 ảnh là Apple Scab
- Precision = 90/100 = 0.9
```

**3. Recall (Độ phủ)**
```
Recall = TP / (TP + FN)

Ví dụ:
- Có 100 ảnh thực sự là Apple Scab
- Model chỉ tìm được 85 ảnh
- Recall = 85/100 = 0.85
```

**4. F1-Score (Trung bình điều hòa)**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.9 × 0.85) / (0.9 + 0.85)
   = 0.874
```

---



