# HƯỚNG DẪN SỬ DỤNG HỆ THỐNG NHẬN DẠNG BỆNH LÁ CÂY

## 📋 MỤC LỤC
1. [Tổng quan](#tổng-quan)
2. [Cài đặt](#cài-đặt)
3. [Huấn luyện mô hình](#huấn-luyện-mô-hình)
4. [Sử dụng giao diện](#sử-dụng-giao-diện)
5. [So sánh KNN vs SVM](#so-sánh-knn-vs-svm)

---

## 🎯 TỔNG QUAN

Hệ thống sử dụng 2 mô hình Machine Learning để nhận dạng bệnh trên lá cây:

### **KNN (K-Nearest Neighbors)**
- ✅ Training nhanh
- ✅ Đơn giản, dễ hiểu
- ❌ Dự đoán chậm hơn
- ❌ Cần nhiều bộ nhớ

### **SVM (Support Vector Machine)**
- ✅ Dự đoán nhanh
- ✅ Chính xác cao
- ❌ Training lâu hơn
- ❌ Phức tạp hơn

### **Dataset**
- Tổng số ảnh: 54,304
- Train: 43,443 ảnh (80%)
- Test: 10,861 ảnh (20%)
- Số lớp bệnh: 38

---

## 🔧 CÀI ĐẶT

### Yêu cầu:
```bash
pip install tensorflow
pip install scikit-learn
pip install streamlit
pip install matplotlib
pip install seaborn
pip install joblib
pip install pillow
```

---

## 🚀 HUẤN LUYỆN MÔ HÌNH

### Bước 1: Chạy file training
```bash
python a.py
```

### Bước 2: Đợi quá trình hoàn thành
- Trích xuất features: ~5-10 phút
- Training KNN: ~10-30 giây
- Training SVM: ~1-3 phút
- Tạo biểu đồ: ~30 giây

### Bước 3: Kiểm tra kết quả
Sau khi chạy xong, bạn sẽ có:

**Files mô hình:**
- `knn_model.joblib` - Mô hình KNN
- `svm_model.joblib` - Mô hình SVM
- `pca_model.joblib` - Mô hình PCA
- `class_names.json` - Danh sách 38 loại bệnh

**Thư mục results/:**
- `confusion_matrix_knn.png` - Ma trận nhầm lẫn KNN
- `confusion_matrix_svm.png` - Ma trận nhầm lẫn SVM
- `model_comparison.png` - So sánh chi tiết 2 mô hình
- `training_time_comparison.png` - So sánh thời gian training
- `training_report.txt` - Báo cáo chi tiết

---

## 🖥️ SỬ DỤNG GIAO DIỆN

### Option 1: Chỉ dùng KNN (file gốc)
```bash
streamlit run ui1.py
```

### Option 2: So sánh KNN vs SVM (file mới)
```bash
streamlit run ui_compare.py
```

### Giao diện sẽ mở tại:
```
http://localhost:8501
```

---

## 📊 SO SÁNH KNN VS SVM

### **Khi nào dùng KNN?**
✅ Cần training nhanh  
✅ Dataset nhỏ  
✅ Không cần dự đoán real-time  
✅ Muốn giải thích dễ hiểu  

### **Khi nào dùng SVM?**
✅ Cần độ chính xác cao  
✅ Dự đoán real-time (sau khi train)  
✅ Dataset lớn  
✅ Có thời gian training  

### **Ví dụ thực tế:**

**Kịch bản 1: Ứng dụng mobile cho nông dân**
→ Chọn **SVM** vì cần dự đoán nhanh trên điện thoại

**Kịch bản 2: Nghiên cứu, thử nghiệm**
→ Chọn **KNN** vì dễ thử nghiệm nhiều tham số

**Kịch bản 3: Hệ thống chuyên nghiệp**
→ Dùng **cả 2**, nếu kết quả giống nhau → tin cậy cao

---

## 🎨 GIẢI THÍCH BIỂU ĐỒ

### 1. **Confusion Matrix**
```
Đường chéo chính (màu đậm) = Dự đoán ĐÚNG
Các ô khác = Dự đoán SAI
```

**Ví dụ:**
- Ô (Apple_scab, Black_rot) = 15
- → 15 ảnh Apple_scab bị nhầm thành Black_rot

### 2. **Model Comparison**
- **Cột 1**: So sánh Accuracy tổng thể
- **Cột 2**: So sánh Precision, Recall, F1-Score
- **Cột 3**: Accuracy theo từng loại bệnh

### 3. **Training Time**
- Cột thấp = Training nhanh
- KNN thường nhanh hơn SVM

---

## 📝 CÁCH SỬ DỤNG UI

### **Bước 1: Chọn mô hình**
Ở sidebar bên trái:
- **KNN**: Chỉ dùng KNN
- **SVM**: Chỉ dùng SVM
- **So sánh cả 2**: Xem kết quả của cả 2 mô hình

### **Bước 2: Upload ảnh**
- Click "Browse files"
- Chọn ảnh lá cây (.jpg, .jpeg, .png)

### **Bước 3: Xem kết quả**
Hệ thống sẽ hiển thị:
- 🌿 Tên cây
- 🦠 Loại bệnh (tiếng Việt)
- 📊 Độ tin cậy (%)
- 🔎 Top 3 dự đoán có xác suất cao nhất

### **Bước 4: So sánh (nếu chọn "So sánh cả 2")**
- Cột trái: Kết quả KNN
- Cột phải: Kết quả SVM
- Dưới cùng: Kết luận (giống nhau hay khác nhau)

---

## ❓ CÂU HỎI THƯỜNG GẶP

### **Q: Tại sao KNN và SVM cho kết quả khác nhau?**
A: Vì 2 mô hình hoạt động theo cách khác nhau:
- KNN: Dựa vào láng giềng gần nhất
- SVM: Dựa vào đường phân chia tối ưu

### **Q: Mô hình nào chính xác hơn?**
A: Xem file `results/training_report.txt` để biết accuracy của từng mô hình

### **Q: Có thể thêm loại bệnh mới không?**
A: Có, nhưng phải train lại từ đầu với dataset mới

### **Q: Tại sao độ tin cậy thấp?**
A: Có thể do:
- Ảnh mờ, không rõ
- Bệnh hiếm, ít dữ liệu training
- Triệu chứng giống nhiều loại bệnh

---

## 🎓 CHO BÁO CÁO

### **Nội dung cần có:**
1. Giới thiệu bài toán
2. Dataset (80:20 train:test)
3. Phương pháp:
   - MobileNetV2 (Transfer Learning)
   - PCA (1280 → 200 features)
   - KNN và SVM
4. Kết quả:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - Training Time
5. So sánh KNN vs SVM
6. Kết luận và hướng phát triển

### **Biểu đồ cần đính kèm:**
- ✅ Confusion Matrix (2 cái)
- ✅ Model Comparison
- ✅ Training Time Comparison

---

## 📞 HỖ TRỢ

Nếu gặp lỗi:
1. Kiểm tra đã cài đủ thư viện chưa
2. Kiểm tra đã train model chưa (chạy `a.py`)
3. Kiểm tra file model có tồn tại không
4. Xem log lỗi để debug

---

**Chúc bạn thành công! 🎉**
