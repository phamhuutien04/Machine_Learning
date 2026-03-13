"""
HỆ THỐNG NHẬN DẠNG BỆNH LÁ CÂY SỬ DỤNG MACHINE LEARNING
=====================================================

Mô tả: Hệ thống sử dụng MobileNetV2 để trích xuất đặc trưng từ ảnh lá cây,
       sau đó áp dụng PCA để giảm chiều và huấn luyện các mô hình ML
       (KNN, SVM) để phân loại bệnh.

Dataset: PlantVillage - Bộ dữ liệu ảnh lá cây với các loại bệnh khác nhau

Kiến trúc:
- Feature Extractor: MobileNetV2 (pre-trained ImageNet)
- Dimensionality Reduction: PCA (1280 → 200 features)
- Classifiers: KNN, SVM

Tác giả: [Tên sinh viên]
Ngày: [Ngày thực hiện]
"""

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.decomposition import PCA
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

# ======================
# CONFIGURATION
# ======================
TRAIN_DIR = "archive/PlantVillage/train"
VAL_DIR = "archive/PlantVillage/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
PCA_COMPONENTS = 200
RANDOM_STATE = 42

# Tạo thư mục kết quả
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("="*60)
print("HỆ THỐNG NHẬN DẠNG BỆNH LÁ CÂY")
print("="*60)
print(f"Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Kích thước ảnh: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"PCA components: {PCA_COMPONENTS}")
print("="*60)

# ======================
# LOAD DATASET
# ======================
def load_dataset():
    """
    Tải dataset PlantVillage và tiền xử lý
    
    Returns:
        train_ds: Dataset huấn luyện
        val_ds: Dataset validation
        class_names: Danh sách tên các lớp
    """
    print("\n📂 Đang tải dataset...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=RANDOM_STATE
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=RANDOM_STATE
    )
    
    class_names = train_ds.class_names
    
    # Chuẩn hóa pixel values [0,1]
    train_ds = train_ds.map(lambda x, y: (x/255.0, y))
    val_ds = val_ds.map(lambda x, y: (x/255.0, y))
    
    print(f"✅ Số lớp: {len(class_names)}")
    print(f"✅ Tên các lớp: {class_names[:5]}... (hiển thị 5 đầu)")
    
    return train_ds, val_ds, class_names

# ======================
# FEATURE EXTRACTOR
# ======================
def build_extractor():
    """
    Xây dựng mô hình trích xuất đặc trưng sử dụng MobileNetV2
    
    Returns:
        model: Mô hình trích xuất đặc trưng
    """
    print("\n🏗️ Xây dựng Feature Extractor...")
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Đóng băng các layer của base model
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    
    print(f"✅ Feature vector size: {model.output_shape[1]}")
    
    return model

# ======================
# EXTRACT FEATURES
# ======================
def extract_features(dataset, extractor, dataset_name):
    """
    Trích xuất đặc trưng từ dataset
    
    Args:
        dataset: TensorFlow dataset
        extractor: Mô hình trích xuất đặc trưng
        dataset_name: Tên dataset (train/val)
    
    Returns:
        X: Ma trận đặc trưng
        y: Nhãn
    """
    print(f"\n🔍 Trích xuất đặc trưng từ {dataset_name} set...")
    
    X = []
    y = []
    
    total_batches = len(list(dataset))
    
    for i, (images, labels) in enumerate(dataset):
        features = extractor(images, training=False).numpy()
        X.append(features)
        y.append(labels.numpy())
        
        if (i + 1) % 10 == 0:
            print(f"  Đã xử lý: {i+1}/{total_batches} batches")
    
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    
    print(f"✅ Shape: X={X.shape}, y={y.shape}")
    
    return X, y

# ======================
# PCA ANALYSIS
# ======================
def apply_pca(X_train, X_val, n_components=PCA_COMPONENTS):
    """
    Áp dụng PCA để giảm chiều dữ liệu
    
    Args:
        X_train: Dữ liệu huấn luyện
        X_val: Dữ liệu validation
        n_components: Số components PCA
    
    Returns:
        X_train_pca: Dữ liệu train sau PCA
        X_val_pca: Dữ liệu val sau PCA
        pca: Mô hình PCA
    """
    print(f"\n📊 Áp dụng PCA (giảm từ {X_train.shape[1]} → {n_components} features)...")
    
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    
    print(f"✅ Explained variance ratio: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    
    # Lưu mô hình PCA
    dump(pca, "pca_model.joblib")
    
    return X_train_pca, X_val_pca, pca

# ======================
# MODEL TRAINING
# ======================
def train_knn(X_train, y_train, X_val, y_val, class_names):
    """Huấn luyện mô hình K-Nearest Neighbors"""
    print("\n🤖 Huấn luyện KNN...")
    
    # Đo thời gian training
    start_time = time.time()
    
    # Thử nghiệm các giá trị k khác nhau
    k_values = [3, 5, 7, 9, 11]
    best_acc = 0
    best_k = 5
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_val)
        acc = accuracy_score(y_val, pred)
        print(f"  K={k}: Accuracy = {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
    
    # Train với k tốt nhất
    print(f"\n🏆 K tốt nhất: {best_k} (Accuracy: {best_acc:.4f})")
    
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Đánh giá chi tiết
    y_pred = knn_final.predict(X_val)
    
    print(f"⏱️ Thời gian training: {training_time:.2f} giây")
    print("\n📊 KNN Classification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Lưu mô hình
    dump(knn_final, "knn_model.joblib")
    
    return knn_final, best_acc, training_time

def train_svm(X_train, y_train, X_val, y_val, class_names):
    """Huấn luyện mô hình Support Vector Machine"""
    print("\n🤖 Huấn luyện SVM...")
    
    # Đo thời gian training
    start_time = time.time()
    
    svm = LinearSVC(random_state=RANDOM_STATE, max_iter=2000)
    svm.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    y_pred = svm.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    print(f"✅ SVM Accuracy: {acc:.4f}")
    print(f"⏱️ Thời gian training: {training_time:.2f} giây")
    
    print("\n📊 SVM Classification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Lưu mô hình
    dump(svm, "svm_model.joblib")
    
    return svm, acc, training_time

# ======================
# VISUALIZATION
# ======================
def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(results, knn_pred, svm_pred, y_val, class_names, training_times):
    """Tạo biểu đồ so sánh giữa KNN và SVM"""
    
    # Biểu đồ so sánh tổng hợp
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Accuracy comparison
    plt.subplot(1, 3, 1)
    models = list(results.keys())
    accuracies = list(results.values())
    colors = ['#3498db', '#e74c3c']
    
    bars = plt.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    plt.title('So sánh Accuracy giữa KNN và SVM', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Thêm giá trị lên đầu cột
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Precision, Recall, F1-score comparison
    plt.subplot(1, 3, 2)
    
    # Tính toán metrics cho từng model
    knn_precision, knn_recall, knn_f1, _ = precision_recall_fscore_support(y_val, knn_pred, average='weighted')
    svm_precision, svm_recall, svm_f1, _ = precision_recall_fscore_support(y_val, svm_pred, average='weighted')
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    knn_scores = [knn_precision, knn_recall, knn_f1]
    svm_scores = [svm_precision, svm_recall, svm_f1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, knn_scores, width, label='KNN', color='#3498db', alpha=0.7)
    plt.bar(x + width/2, svm_scores, width, label='SVM', color='#e74c3c', alpha=0.7)
    
    plt.title('So sánh Precision, Recall, F1-Score', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    plt.xlabel('Metrics')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Subplot 3: Accuracy per class (Top 10)
    plt.subplot(1, 3, 3)
    
    # Tính accuracy per class
    knn_cm = confusion_matrix(y_val, knn_pred)
    svm_cm = confusion_matrix(y_val, svm_pred)
    
    knn_class_acc = np.diag(knn_cm) / np.sum(knn_cm, axis=1)
    svm_class_acc = np.diag(svm_cm) / np.sum(svm_cm, axis=1)
    
    # Chỉ hiển thị top 10 classes có accuracy cao nhất
    top_classes = np.argsort(knn_class_acc + svm_class_acc)[-10:]
    
    x = np.arange(len(top_classes))
    plt.bar(x - width/2, knn_class_acc[top_classes], width, label='KNN', color='#3498db', alpha=0.7)
    plt.bar(x + width/2, svm_class_acc[top_classes], width, label='SVM', color='#e74c3c', alpha=0.7)
    
    plt.title('Accuracy theo từng lớp (Top 10)', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Classes')
    plt.xticks(x, [class_names[i][:15] + '...' if len(class_names[i]) > 15 else class_names[i] 
                   for i in top_classes], rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_time_comparison(training_times):
    """Biểu đồ so sánh thời gian huấn luyện"""
    plt.figure(figsize=(8, 6))
    
    models = list(training_times.keys())
    times = list(training_times.values())
    colors = ['#3498db', '#e74c3c']
    
    bars = plt.bar(models, times, color=colors, alpha=0.7, edgecolor='black')
    
    plt.title('So sánh thời gian huấn luyện', fontsize=14, fontweight='bold')
    plt.ylabel('Thời gian (giây)')
    plt.xlabel('Mô hình')
    
    # Thêm giá trị lên đầu cột
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(results, training_times, class_names):
    """Tạo báo cáo tổng kết"""
    report = f"""
HỆ THỐNG NHẬN DẠNG BỆNH LÁ CÂY - BÁO CÁO KẾT QUẢ
================================================

Thời gian thực hiện: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

THÔNG TIN DATASET:
- Tổng số lớp: {len(class_names)}
- Kích thước ảnh: {IMG_SIZE}
- PCA components: {PCA_COMPONENTS}

KẾT QUẢ HUẤN LUYỆN:
"""
    
    for model_name, accuracy in results.items():
        training_time = training_times[model_name]
        report += f"- {model_name}: Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%), Training time = {training_time:.2f}s\n"
    
    best_model = max(results, key=results.get)
    fastest_model = min(training_times, key=training_times.get)
    
    report += f"\nMÔ HÌNH CHÍNH XÁC NHẤT: {best_model} - {results[best_model]:.4f}\n"
    report += f"MÔ HÌNH NHANH NHẤT: {fastest_model} - {training_times[fastest_model]:.2f}s\n"
    
    report += f"""
DANH SÁCH CÁC LỚP BỆNH:
{chr(10).join([f"{i+1:2d}. {name}" for i, name in enumerate(class_names)])}

FILES ĐÃ TẠO:
- knn_model.joblib: Mô hình KNN
- svm_model.joblib: Mô hình SVM  
- pca_model.joblib: Mô hình PCA
- class_names.json: Danh sách tên lớp
- results/: Thư mục chứa biểu đồ và báo cáo
  + confusion_matrix_knn.png: Ma trận nhầm lẫn KNN
  + confusion_matrix_svm.png: Ma trận nhầm lẫn SVM
  + model_comparison.png: So sánh chi tiết 2 mô hình
  + training_time_comparison.png: So sánh thời gian huấn luyện
"""
    
    with open("results/training_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n" + "="*60)
    print(report)
    print("="*60)

# ======================
# MAIN EXECUTION
# ======================
def main():
    """Hàm chính thực hiện toàn bộ quy trình"""
    
    # 1. Load dataset
    train_ds, val_ds, class_names = load_dataset()
    
    # 2. Build feature extractor
    extractor = build_extractor()
    
    # 3. Extract features
    X_train, y_train = extract_features(train_ds, extractor, "train")
    X_val, y_val = extract_features(val_ds, extractor, "validation")
    
    # 4. Apply PCA
    X_train_pca, X_val_pca, pca = apply_pca(X_train, X_val)
    
    # 5. Train models
    results = {}
    training_times = {}
    
    knn_model, knn_acc, knn_time = train_knn(X_train_pca, y_train, X_val_pca, y_val, class_names)
    results["KNN"] = knn_acc
    training_times["KNN"] = knn_time
    
    svm_model, svm_acc, svm_time = train_svm(X_train_pca, y_train, X_val_pca, y_val, class_names)
    results["SVM"] = svm_acc
    training_times["SVM"] = svm_time
    
    # 6. Create visualizations
    print("\n📊 Tạo biểu đồ...")
    
    # Get predictions for comparison
    knn_pred = knn_model.predict(X_val_pca)
    svm_pred = svm_model.predict(X_val_pca)
    
    # Confusion matrices
    plot_confusion_matrix(y_val, knn_pred, class_names, "KNN")
    plot_confusion_matrix(y_val, svm_pred, class_names, "SVM")
    
    # Model comparison charts
    plot_model_comparison(results, knn_pred, svm_pred, y_val, class_names, training_times)
    
    # Training time comparison
    plot_training_time_comparison(training_times)
    
    # 7. Save class names
    with open("class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    
    # 8. Create summary report
    create_summary_report(results, training_times, class_names)
    
    print(f"\n🎉 Hoàn thành! Thời gian kết thúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()