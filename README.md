# 🌾 Nutrisi-Padi

Aplikasi berbasis **Python + Flask** untuk mendeteksi **kekurangan nutrisi pada tanaman padi** menggunakan **model CNN** dan pencocokan **gejala** dari file CSV.

---

## 🧠 Fitur Utama

- 🔍 **Deteksi visual** kekurangan nutrisi dari gambar daun padi menggunakan model CNN.
- 📝 **Pencarian manual** berdasarkan gejala dan fase pertumbuhan tanaman padi.
- 💻 **Antarmuka web interaktif** berbasis Flask yang ringan dan mudah digunakan.

---

## 🚀 Cara Menjalankan (Local)

### 1. Clone repositori
```bash
git clone https://github.com/MuhamadFahriSalam/Nutrisi-Padi.git
cd Nutrisi-Padi
```

### 2. Buat environment baru
```bash
python -m venv env
env\Scripts\activate     # Gunakan 'source env/bin/activate' di Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Latih dan ekspor model CNN
Jalankan script di bawah untuk melatih model dan menyimpan file:
- `rice_leaf_diseases.keras` (model CNN)
- `label_map.json` (label klasifikasi)

```bash
python train_and_export_model.py
```

### 5. Jalankan aplikasi Flask
```bash
python app.py
```

### 6. Akses melalui browser
Buka:
```
http://localhost:5000
```

---

## 📁 Struktur File Penting

| File / Folder                | Deskripsi                                                                 |
|-----------------------------|--------------------------------------------------------------------------|
| `app.py`                    | Aplikasi Flask utama                                                     |
| `train_and_export_model.py` | Script pelatihan dan ekspor model CNN                                    |
| `rice_leaf_diseases.keras`  | Model CNN hasil pelatihan (otomatis dibuat)                              |
| `label_map.json`            | Label indeks ke nama kelas (otomatis dibuat)                             |
| `kekurangan_nutrisi_padi_bersih.csv` | Dataset gejala dan fase pertumbuhan tanaman                        |
| `templates/`                | Template HTML Flask                                                      |
| `static/`                   | Folder untuk menyimpan gambar yang diunggah                              |

---

## 📸 Contoh Penggunaan

1. Upload gambar daun padi untuk deteksi otomatis.
2. Atau masukkan deskripsi gejala dan fase pertumbuhan untuk pencarian manual.
3. Hasil deteksi/diagnosis akan muncul di bawah input.

---

## 📌 Catatan

- Pastikan dataset gambar terstruktur dalam subfolder per kelas sebelum pelatihan.
- Model akan otomatis dibuat jika belum tersedia saat menjalankan aplikasi.
