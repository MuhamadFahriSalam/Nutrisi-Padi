from flask import Flask, request, render_template, jsonify
import pandas as pd
import os, json
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
os.makedirs('static', exist_ok=True)

# Load model dan label
model = load_model("rice_leaf_diseases.keras") # Pastikan model sudah dilatih dan disimpan sebelumnya
with open("label_map.json") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}

# Load CSV data deskripsi defisiensi
df = pd.read_csv('kekurangan_nutrisi_padi_bersih.csv').reset_index(drop=True) # Pastikan file CSV ada di direktori yang sama dan formatnya

# Mapping hasil label model ke nama defisiensi di CSV
label_to_defisiensi = {
    "Nitrogen(N)": "Kekurangan Nitrogen (N)",
    "Phosphorus(P)": "Kekurangan Fosfor (P)",
    "Potassium(K)": "Kekurangan Kalium (K)",
    "Magnesium(Mg)": "Kekurangan Magnesium (Mg)",
    "Calcium(Ca)": "Kekurangan Kalsium (Ca)",
    "Sulfur(S)": "Kekurangan Sulfur (S)",
    "Iron(Fe)": "Kekurangan Zat Besi (Fe)",
    "Manganese(Mn)": "Kekurangan Mangan (Mn)",
    "Zinc(Zn)": "Kekurangan Seng (Zn)",
    "Boron(B)": "Kekurangan Boron (B)",
    "Copper(Cu)": "Kekurangan Tembaga (Cu)",
    "Molybdenum(Mo)": "Kekurangan Molibdenum (Mo)",
    "Chlorophyll": "Kekurangan Klorofil (Chlorophyll)",
    "Amino Acid": "Kekurangan Asam Amino (Amino Acid)",
    "Organic Matter": "Kekurangan Zat Organik (Organic Matter)",
    "Water Stress": "Kekurangan Air (Water Stress)",
    "Light Stress": "Kekurangan Cahaya (Light Stress)",
    "Temperature Stress": "Kekurangan Suhu (Temperature Stress)",
    "Soil pH": "Kekurangan pH Tanah (Soil pH)",
    "Microbial Deficiency": "Kekurangan Mikroba (Microbial Deficiency)",
    "Micronutrient Deficiency": "Kekurangan Unsur Mikro (Micronutrient Deficiency)",
    "Macronutrient Deficiency": "Kekurangan Unsur Makro (Macronutrient Deficiency)",
    "Nutrient Deficiency": "Kekurangan Unsur Hara (Nutrient Deficiency)",
    "Specific Nutrient Deficiency": "Kekurangan Unsur Hara Spesifik (Specific Nutrient Deficiency)",
    "Normal": "Normal"
}

@app.route('/')
def index():
    return render_template('index.html') # Halaman utama untuk mengunggah gambar

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files.get('image')
    if not img_file:
        return jsonify({'error': 'Tidak ada gambar yang diunggah'}), 400

    filename = secure_filename(img_file.filename)
    filepath = os.path.join('static', filename)
    img_file.save(filepath)

    try:
        # Preprocessing gambar
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi dengan model
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        label_pred = idx_to_label[predicted_index]

        # Mapping ke defisiensi di CSV
        defisiensi_key = label_to_defisiensi.get(label_pred, None)
        if defisiensi_key:
            row = df[df['Defisiensi Nutrisi'] == defisiensi_key]
        else:
            row = pd.DataFrame()

        # Ambil data hasil atau fallback
        if not row.empty:
            result = row.iloc[0].to_dict()
        else:
            result = {
                'Defisiensi Nutrisi': label_pred,
                'Metode Pengendalian': 'Tidak ditemukan.',
                'Produk Nutrisi yang Direkomendasikan': 'Tidak ditemukan.'
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Gagal memproses gambar: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
