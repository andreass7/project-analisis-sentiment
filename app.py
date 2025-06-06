import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
from wordcloud import WordCloud
from collections import Counter
import nltk
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from preprocessing import cleaningText, casefoldingText, tokenizingText, filteringText, toSentence, fix_slangwords, stemmingText


# Config awal halaman
st.set_page_config(page_title="Analisis Sentimen", layout="wide")

# Sidebar
st.sidebar.title("📚 Navigasi Aplikasi")
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Dashboard", "📝 Analisis Teks Baru"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Tentang Aplikasi**")
st.sidebar.info("Aplikasi ini melakukan analisis sentimen pada teks menggunakan model Naive Bayes dan KNN dengan preprocessing bahasa Indonesia.")

# Load data
preprocessed_df = pd.read_csv("data/df_cleaned.csv")
with open("model/metrics.json") as f:
    results = json.load(f)
with open("model/confusion_data.json") as f:
    confusion_data = json.load(f)
df_results = pd.DataFrame(results)
df_results.rename(columns={"index": "split"}, inplace=True)

# Label Mapping
label_map = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}

# Fungsi untuk mengunduh NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt_tab')

# Panggil fungsi ini saat awal streamlit
download_nltk_resources()


# Dashboard
if page == "🏠 Dashboard":
    st.title("📊 Dashboard Analisis Sentimen")

    # WordCloud
    st.subheader("☁️ WordCloud berdasarkan Sentimen")
    col1, col2, col3 = st.columns(3)
    label_emoji = {"negatif": "😠", "netral": "😐", "positif": "😊"}
    for label in ["negatif", "netral", "positif"]:
        st.markdown(f"### {label_emoji[label]} {label.capitalize()}")
        filtered_df = preprocessed_df[preprocessed_df['label'] == label]
        text_data = " ".join(filtered_df['text_akhir'])
        if text_data.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
            st.image(wordcloud.to_array(), use_container_width=True)
        else:
            st.info(f"Tidak ada data untuk label **{label}**.")
    
    # KATA SERING MUNCUL 
    st.subheader("🔁 20 Kata Paling Sering Muncul (Semua Data)")
    tokens = " ".join(preprocessed_df['text_akhir']).split()
    word_freq = Counter(tokens)
    most_common_df = pd.DataFrame(word_freq.most_common(20), columns=["Kata", "Frekuensi"])
    st.dataframe(most_common_df)

    st.subheader("📊 Evaluasi Model per Split")

    # Pilih split dan model
    split_choice = st.selectbox("Pilih Split Data", df_results['split'].unique())
    model_choice = st.selectbox("Pilih Model", df_results['model'].unique())

    # Filter berdasarkan pilihan
    selected = df_results[(df_results['split'] == split_choice) & (df_results['model'] == model_choice)]

    # Tampilkan metrik evaluasi
    st.markdown(f"""
    - 🎯 **Akurasi (Test):** {selected['Test_Accuracy'].values[0]:.2%}
    - 🎯 **Akurasi (CV):** {selected['CV_Accuracy'].values[0]:.2%}
    - 📏 **Precision:** {selected['Precision'].values[0]:.2%}
    - 🔁 **Recall:** {selected['Recall'].values[0]:.2%}
    - 📦 **F1 Score:** {selected['F1_Score'].values[0]:.2%}
    - 🔢 **Ukuran Data Train:** {selected['train_size'].values[0]}
    - 🔢 **Ukuran Data Test:** {selected['test_size'].values[0]}
    """)

    if st.button("Tampilkan Confusion Matrix"):
        y_true = confusion_data[split_choice][model_choice]["y_true"]
        y_pred = confusion_data[split_choice][model_choice]["y_pred"]

        labels = [0, 1, 2]  # label asli dalam bentuk angka
        display_labels = [label_map[l] for l in labels]

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f"Confusion Matrix {model_choice} - Split {split_choice.replace('_', ':')}")
        st.pyplot(fig)


        
# Analisis Teks
elif page == "📝 Analisis Teks Baru":
    st.title("🧠 Analisis Sentimen pada Teks")
    st.markdown("Masukkan teks, pilih model, dan lihat hasil prediksi beserta proses preprocessing.")

    input_text = st.text_area("📥 Masukkan Teks:")
    selected_model = st.selectbox("📌 Pilih Model", ["Naive Bayes", "KNN"])

    if st.button("🔍 Proses Analisis"):
        # Preprocessing
        cleaned = cleaningText(input_text)
        folded = casefoldingText(cleaned)
        slang_fixed = fix_slangwords(folded)
        tokens = tokenizingText(slang_fixed)
        filtered = filteringText(tokens)
        stemmed = stemmingText(toSentence(filtered))

        st.markdown("### 🧼 Hasil Preprocessing")
        st.code(f"Cleaned: {cleaned}")
        st.code(f"Casefolded: {folded}")
        st.code(f"Slangword Fixed: {slang_fixed}")
        st.code(f"Tokens: {tokens}")
        st.code(f"Stopwords Removed: {filtered}")
        st.code(f"Stemmed: {stemmed}")

        # Load model
        model_path = "model/nb_pipeline.pkl" if selected_model == "Naive Bayes" else "model/knn_pipeline.pkl"
        model = joblib.load(model_path)

        # Predict
        pred = model.predict([stemmed])[0]
        label = label_map.get(pred, "Tidak diketahui")

        st.markdown("### 📢 Hasil Klasifikasi")
        st.success(f"Hasil Prediksi: **{label}**")
