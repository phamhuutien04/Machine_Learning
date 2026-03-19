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

MODEL_DIR = "models"

KNN_MODEL_PATH = f"{MODEL_DIR}/knn_model.joblib"
SVM_MODEL_PATH = f"{MODEL_DIR}/svm_model.joblib"
SCALER_PATH = f"{MODEL_DIR}/scaler.joblib"
CLASSES_PATH = f"{MODEL_DIR}/class_names.json"

st.set_page_config(
    page_title="Plant Disease Detection - KNN vs SVM",
    page_icon="🍃",
    layout="wide"
)

# =========================
# VI MAP
# =========================
VI_MAP = {
"Apple_scab":"Bệnh ghẻ táo",
"Black_rot":"Bệnh thối đen",
"Cedar_apple_rust":"Bệnh gỉ sắt táo",
"healthy":"Khỏe mạnh",
"Powdery_mildew":"Bệnh phấn trắng",
"Early_blight":"Bệnh cháy lá sớm",
"Late_blight":"Bệnh cháy lá muộn"
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


def preprocess_image(img):
    if img.mode!="RGB":
        img=img.convert("RGB")

    img=img.resize(IMG_SIZE)
    arr=np.array(img).astype("float32")/255.0
    return np.expand_dims(arr,axis=0)


# =========================
# FEATURE EXTRACTOR
# =========================
@st.cache_resource
def build_feature_extractor():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable=False

    return tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D()
    ])


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    knn = load(KNN_MODEL_PATH)
    svm = load(SVM_MODEL_PATH)
    scaler = load(SCALER_PATH)

    with open(CLASSES_PATH,"r",encoding="utf-8") as f:
        class_names=json.load(f)

    return knn, svm, scaler, class_names


# =========================
# PREDICT
# =========================
def predict_knn(extractor, knn, scaler, img_arr):
    feat = extractor(img_arr,training=False).numpy()
    feat = scaler.transform(feat)

    pred_idx=int(knn.predict(feat)[0])
    probs=knn.predict_proba(feat)[0]
    conf=float(np.max(probs))

    top_idx=np.argsort(probs)[::-1][:K_TOP]
    top_list=[(int(i),float(probs[i])) for i in top_idx]

    return pred_idx, conf, top_list


def predict_svm(extractor, svm, scaler, img_arr):
    feat = extractor(img_arr,training=False).numpy()
    feat = scaler.transform(feat)

    pred_idx=int(svm.predict(feat)[0])

    decision = svm.decision_function(feat)[0]
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
st.sidebar.write("Model: MobileNetV2 + Scaler")
st.sidebar.write("Input size:",IMG_SIZE)
st.sidebar.write("Top predictions:",K_TOP)

st.sidebar.markdown("---")
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
# LOAD
# =========================
try:
    knn, svm, scaler, class_names = load_models()
    extractor = build_feature_extractor()
except Exception as e:
    st.error("❌ Không load được model")
    st.code(str(e))
    st.stop()

# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader("📤 Upload ảnh lá", type=["jpg","jpeg","png"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Ảnh đã upload", use_container_width=True)

    img_arr = preprocess_image(img)

    with st.spinner("🔍 Đang phân tích..."):

        if model_choice == "KNN":
            pred_idx, conf, top_list = predict_knn(extractor, knn, scaler, img_arr)

            label = class_names[pred_idx]
            _, disease = parse_label(label)

            st.success("✅ Kết quả dự đoán - KNN")

            col1, col2, col3 = st.columns(3)
            col2.metric("🦠 Bệnh", disease)
            col3.metric("📊 Độ tin cậy", f"{conf*100:.2f}%")

            st.progress(conf)

            st.subheader("🔎 Top 3 dự đoán")
            medals = ["🥇","🥈","🥉"]
            for i, (idx, p) in enumerate(top_list):
                lbl = class_names[idx]
                pl, dis = parse_label(lbl)
                st.markdown(f"{medals[i]} **{pl} — {dis}**  \n📊 {p*100:.2f}%")

        elif model_choice == "SVM":
            pred_idx, conf, top_list = predict_svm(extractor, svm, scaler, img_arr)

            label = class_names[pred_idx]
            _, disease = parse_label(label)

            st.success("✅ Kết quả dự đoán - SVM")

            col1, col2, col3 = st.columns(3)
            col2.metric("🦠 Bệnh", disease)
            col3.metric("📊 Độ tin cậy", f"{conf*100:.2f}%")

            st.progress(conf)

            st.subheader("🔎 Top 3 dự đoán")
            medals = ["🥇","🥈","🥉"]
            for i, (idx, p) in enumerate(top_list):
                lbl = class_names[idx]
                pl, dis = parse_label(lbl)
                st.markdown(f"{medals[i]} **{pl} — {dis}**  \n📊 {p*100:.2f}%")

        else:
            knn_pred, knn_conf, knn_top = predict_knn(extractor, knn, scaler, img_arr)
            svm_pred, svm_conf, svm_top = predict_svm(extractor, svm, scaler, img_arr)

            st.success("✅ So sánh KNN vs SVM")

            col_knn, col_svm = st.columns(2)

            medals = ["🥇","🥈","🥉"]

            with col_knn:
                st.markdown("### 🔵 KNN")
                _, dis = parse_label(class_names[knn_pred])
                st.metric("🦠 Bệnh", dis)
                st.metric("📊 Độ tin cậy", f"{knn_conf*100:.2f}%")
                st.progress(knn_conf)

                st.markdown("**Top 3:**")
                for i, (idx, p) in enumerate(knn_top):
                    lbl = class_names[idx]
                    pl, d = parse_label(lbl)
                    st.markdown(f"{medals[i]} **{pl} — {d}**  \n📊 {p*100:.2f}%")

            with col_svm:
                st.markdown("### 🔴 SVM")
                _, dis = parse_label(class_names[svm_pred])
                st.metric("🦠 Bệnh", dis)
                st.metric("📊 Độ tin cậy", f"{svm_conf*100:.2f}%")
                st.progress(svm_conf)

                st.markdown("**Top 3:**")
                for i, (idx, p) in enumerate(svm_top):
                    lbl = class_names[idx]
                    pl, d = parse_label(lbl)
                    st.markdown(f"{medals[i]} **{pl} — {d}**  \n📊 {p*100:.2f}%")

            st.markdown("---")

            if knn_pred == svm_pred:
                st.success("🤖 Cả 2 model dự đoán giống nhau")
            else:
                st.warning("⚠️ Kết quả khác nhau giữa 2 model")

else:
    st.info("📷 Hãy upload ảnh lá cây để bắt đầu")