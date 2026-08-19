# 📈 Sales Prediction ML System  

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
![XGBoost](https://img.shields.io/badge/XGBoost-4285F4?style=for-the-badge&logo=xgboost&logoColor=white)  

**Built By Stanley**  

A fully interactive Machine Learning web application that predicts **future product sales** based on historical sales data. The model has been trained with an XGBoost regressor and a custom preprocessing pipeline, then packaged for instant inference via Streamlit.

---

## 🚀 Live Demo  
**Try the app here:** [Sales Prediction ML System](https://sale-prediction-ml-system.streamlit.app/)

---

## 📖 Project Overview  
Accurate sales forecasting is critical for inventory planning, budgeting, and marketing strategy. This project using the `sales_prediction_task2.csv` dataset demonstrates how to:

1. **Load** a pre‑trained XGBoost model and its associated `ColumnTransformer`.  
2. **Capture** user inputs (date, product category, price, promotion, discounts, etc.) through a clean Streamlit UI.  
3. **Engineer** derived features on the fly (e.g., final price after discount).  
4. **Predict** the expected sales amount in real‑time and display the result with a friendly message.

The underlying model achieved the following performance on the hold‑out set:

| Metric | Value |
|--------|-------|
| **MAE** | **279.28** |
| **MSE** | **128803.21** |
| **RMSE** | **358.89** |
| **R²** | **0.8113** |

---

## 🛠️ Technology Stack  

* **Language**: Python  
* **Data Processing**: pandas, NumPy  
* **Machine Learning**: XGBoost, scikit‑learn (`ColumnTransformer`), joblib for serialization  
* **Web Framework**: Streamlit  
* **Deployment**: Streamlit Community Cloud (public URL above)  

---

## 📂 Repository Structure  

```
/ (repo root)
│
├─ streamlit_app.py               # Streamlit UI + inference logic
├─ requirements.txt                # Exact Python dependencies
├─ xgboost_sales_model.pkl        # Trained XGBoost regressor (binary)
├─ preprocessor.pkl                # Saved ColumnTransformer pipeline
├─ sales_prediction_task2.csv     # Original dataset used for training
└─ README.md                      # This documentation
```

*All files are placed in the repository root so that Streamlit can locate the model assets directly when the app starts.*

---

## 💻 How to Run Locally  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<YOUR‑USERNAME>/<YOUR‑REPO>.git
   cd <YOUR‑REPO>
   ```

2. **Create a virtual environment (optional but recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app**  
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser** – Streamlit will automatically open `http://localhost:8501` (or you can navigate there manually).

---

## 🧩 How It Works (inside `streamlit_app.py`)  

1. **Asset Loading** – `load_assets()` caches the `preprocessor.pkl` and `xgboost_sales_model.pkl` using `@st.cache_resource`.  
2. **User Input UI** – Two columns collect:
   * Date → month, day‑of‑week, week‑of‑year, year  
   * Product details (category, price, promotion checkbox, discount slider, quantity)  
   * Historical sales features (previous‑day, previous‑week, rolling‑7‑day)  
3. **Feature Engineering** – If `DiscountRate` and `Price` are provided, a derived column `FinalPrice = Price * (1‑DiscountRate)` is added (matching the training pipeline).  
4. **Prediction** – On button click, the input DataFrame is transformed with the cached `ColumnTransformer` and fed to the XGBoost model. The resulting sales estimate is shown with `st.success`.  

---

## 📦 Deploying to Streamlit Community Cloud (the simplest method)

1. **Push all files** (including the two `.pkl` files) to a GitHub repository.  
2. **Navigate to** [https://share.streamlit.io/](https://share.streamlit.io/) and click **“New app”**.  
3. **Select** your repository, the branch (e.g., `main`), and the file `streamlit_app.py`.  
4. Click **Deploy** – Streamlit will install the dependencies from `requirements.txt`, download the model artifacts, and give you a public URL (the one shown above).  

Any subsequent `git push` will automatically trigger a rebuild and update the live app.

---

## 🛡️ Known Limitations & Future Enhancements  

* **Category handling** – The `OneHotEncoder` in the preprocessor uses `handle_unknown='ignore'`. New product categories will be silently ignored (treated as all zeros).  
* **Scalability** – For very high‑traffic usage you may want to move inference to a separate API (FastAPI, Flask) and call it from Streamlit to decouple UI from model serving.  
* **Feature expansion** – Adding more temporal features (seasonality flags, holidays) could push the R² above 0.85.  
* **Model versioning** – Store model assets in a version‑controlled bucket (e.g., AWS S3) and load them dynamically to enable A/B testing of new models.

---

## 📫 Contact  

Developed by **Stanley**. Feel free to open an issue or submit a pull request if you have suggestions, bug reports, or want to contribute new features!
