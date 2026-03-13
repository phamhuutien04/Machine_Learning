import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from joblib import load

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
K_TOP = 3

KNN_MODEL_PATH = "knn_model.joblib"
SVM_MODEL_PATH = "svm_model.joblib"
PCA_PATH = "pca_model.joblib"
CLASSES_PATH = "class_names.json"

st.set_page_config(
    page_title="Plant Disease Detection - KNN vs SVM",
    page_icon="🍃",
    layout="wide"
)

# =========================
# VIETNAMESE DICTIONARY
# =========================
VI_MAP = {
"Apple_scab":"Bệnh ghẻ táo",
"Black_rot":"Bệnh thối đen",
"Cedar_apple_rust":"Bệnh gỉ sắt táo",
"healthy":"Khỏe mạnh",
"Powdery_mildew":"Bệnh phấn trắng",
"Cercospora_leaf_spot Gray_leaf_spot":"Bệnh đốm lá xám",
"Common_rust_":"Bệnh gỉ sắt",
"Northern_Leaf_Blight":"Bệnh cháy lá",
"Esca_(Black_Measles)":"Bệnh Esca",
"Leaf_blight_(Isariopsis_Leaf_Spot)":"Bệnh cháy lá",
"Haunglongbing_(Citrus_greening)":"Bệnh vàng lá gân xanh",
"Bacterial_spot":"Bệnh đốm vi khuẩn",
"Early_blight":"Bệnh cháy lá sớm",
"Late_blight":"Bệnh cháy lá muộn",
"Leaf_Mold":"Bệnh mốc lá",
"Septoria_leaf_spot":"Bệnh đốm lá Septoria",
"Spider_mites Two-spotted_spider_mite":"Nhện đỏ hai chấm",
"Target_Spot":"Bệnh đốm mục tiêu",
"Tomato_Yellow_Leaf_Curl_Virus":"Virus xoăn lá vàng",
"Tomato_mosaic_virus":"Virus khảm"
}

# =========================
# UTILS
# =========================
def parse_label(label):
    if "___" in label:
        plant, disease = label.split("___",1)
    else:
        plant, disease = "Unknown", label

    plant = plant.replace("_"," ").replace(",","").replace("(","").replace(")","")
    disease_vi = VI_MAP.get(disease, disease.replace("_"," "))

    return plant, disease_vi


@st.cache_resource
def load_models():
    knn = load(KNN_MODEL_PATH)
    svm = load(SVM_MODEL_PATH)
    pca = load(PCA_PATH)

    with open(CLASSES_PATH,"r",encoding="utf-8") as f:
        class_names=json.load(f)

    return knn, svm, pca, class_names


@st.cache_resource
def build_feature_extractor():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable=False

    model=tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    return model


def preprocess_image(img):
    if img.mode!="RGB":
        img=img.convert("RGB")

    img=img.resize(IMG_SIZE)
    arr=np.array(img).astype("float32")/255.0
    arr=np.expand_dims(arr,axis=0)

    return arr


def predict_knn(extractor, knn, pca, img_arr):
    feat = extractor(img_arr,training=False).numpy()
    feat = pca.transform(feat)

    pred_idx=int(knn.predict(feat)[0])
    probs=knn.predict_proba(feat)[0]
    conf=float(np.max(probs))

    top_idx=np.argsort(probs)[::-1][:K_TOP]
    top_list=[(int(i),float(probs[i])) for i in top_idx]

    return pred_idx, conf, top_list


def predict_svm(extractor, svm, pca, img_arr):
    feat = extractor(img_arr,training=False).numpy()
    feat = pca.transform(feat)

    pred_idx=int(svm.predict(feat)[0])
    
    # SVM không có predict_proba, dùng decision_function
    decision = svm.decision_function(feat)[0]
    
    # Chuyển decision scores thành xác suất đơn giản
    exp_scores = np.exp(decision - np.max(decision))
    probs = exp_scores / np.sum(exp_scores)
    
    conf=float(np.max(probs))

    top_idx=np.argsort(probs)[::-1][:K_TOP]
    top_list=[(int(i),float(probs[i])) for i in top_idx]

    return pred_idx, conf, top_list


# =========================
# SIDEBAR
# =========================
st.sidebar.title("🍃 Plant Disease AI")
st.sidebar.write("Model: MobileNetV2 + PCA")
st.sidebar.write("Input size:",IMG_SIZE)
st.sidebar.write("Top predictions:",K_TOP)

st.sidebar.markdown("---")
st.sidebar.subheader("Chọn mô hình:")
model_choice = st.sidebar.radio(
    "Mô hình phân loại:",
    ["KNN", "SVM", "So sánh cả 2"],
    index=2
)

# =========================
# MAIN UI
# =========================
st.title("🍃 Hệ thống nhận diện bệnh lá cây")
st.caption("Upload ảnh lá cây để hệ thống AI dự đoán bệnh.")

# =========================
# LOAD MODEL
# =========================
try:
    knn, svm, pca, class_names = load_models()
    extractor = build_feature_extractor()
except Exception as e:
    st.error("❌ Không load được model")
    st.code(str(e))
    st.stop()

# =========================
# FILE UPLOAD
# =========================
uploaded = st.file_uploader(
    "📤 Upload ảnh lá",
    type=["jpg","jpeg","png"]
)

# =========================
# PREDICTION
# =========================
if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Ảnh đã upload", use_container_width=True)

    img_arr = preprocess_image(img)

    with st.spinner("🔍 AI đang phân tích..."):
        if model_choice == "KNN":
            # Chỉ dự đoán KNN
            pred_idx, conf, top_list = predict_knn(extractor, knn, pca, img_arr)
            
            label = class_names[pred_idx]
            plant, disease = parse_label(label)

            st.success("✅ Kết quả dự đoán - KNN")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🌿 Tên cây", plant)
            col2.metric("🦠 Bệnh", disease)
            col3.metric("📊 Độ tin cậy", f"{conf*100:.2f}%")
            
            st.progress(conf)

            st.subheader("🔎 Top dự đoán")
            medals = ["🥇","🥈","🥉"]
            for i, (idx, p) in enumerate(top_list):
                lbl = class_names[idx]
                pl, dis = parse_label(lbl)
                st.write(f"{medals[i]} **{pl} — {dis}**  |  {p*100:.2f}%")

        elif model_choice == "SVM":
            # Chỉ dự đoán SVM
            pred_idx, conf, top_list = predict_svm(extractor, svm, pca, img_arr)
            
            label = class_names[pred_idx]
            plant, disease = parse_label(label)

            st.success("✅ Kết quả dự đoán - SVM")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🌿 Tên cây", plant)
            col2.metric("🦠 Bệnh", disease)
            col3.metric("📊 Độ tin cậy", f"{conf*100:.2f}%")
            
            st.progress(conf)

            st.subheader("🔎 Top dự đoán")
            medals = ["🥇","🥈","🥉"]
            for i, (idx, p) in enumerate(top_list):
                lbl = class_names[idx]
                pl, dis = parse_label(lbl)
                st.write(f"{medals[i]} **{pl} — {dis}**  |  {p*100:.2f}%")

        else:
            # So sánh cả 2 mô hình
            knn_pred_idx, knn_conf, knn_top = predict_knn(extractor, knn, pca, img_arr)
            svm_pred_idx, svm_conf, svm_top = predict_svm(extractor, svm, pca, img_arr)

            st.success("✅ So sánh kết quả KNN vs SVM")

            # Tạo 2 cột để so sánh
            col_knn, col_svm = st.columns(2)

            with col_knn:
                st.markdown("### 🔵 KNN")
                knn_label = class_names[knn_pred_idx]
                knn_plant, knn_disease = parse_label(knn_label)
                
                st.metric("🌿 Tên cây", knn_plant)
                st.metric("🦠 Bệnh", knn_disease)
                st.metric("📊 Độ tin cậy", f"{knn_conf*100:.2f}%")
                st.progress(knn_conf)

                st.markdown("**Top 3 dự đoán:**")
                medals = ["🥇","🥈","🥉"]
                for i, (idx, p) in enumerate(knn_top):
                    lbl = class_names[idx]
                    pl, dis = parse_label(lbl)
                    st.write(f"{medals[i]} {pl} — {dis} ({p*100:.1f}%)")

            with col_svm:
                st.markdown("### 🔴 SVM")
                svm_label = class_names[svm_pred_idx]
                svm_plant, svm_disease = parse_label(svm_label)
                
                st.metric("🌿 Tên cây", svm_plant)
                st.metric("🦠 Bệnh", svm_disease)
                st.metric("📊 Độ tin cậy", f"{svm_conf*100:.2f}%")
                st.progress(svm_conf)

                st.markdown("**Top 3 dự đoán:**")
                medals = ["🥇","🥈","🥉"]
                for i, (idx, p) in enumerate(svm_top):
                    lbl = class_names[idx]
                    pl, dis = parse_label(lbl)
                    st.write(f"{medals[i]} {pl} — {dis} ({p*100:.1f}%)")

            # Kết luận
            st.markdown("---")
            if knn_pred_idx == svm_pred_idx:
                st.info(f"✅ **Cả 2 mô hình đều dự đoán giống nhau:** {knn_disease}")
            else:
                st.warning(f"⚠️ **Kết quả khác nhau!** KNN: {knn_disease} | SVM: {svm_disease}")

else:
    st.info("📷 Hãy upload ảnh lá cây để bắt đầu dự đoán.")
