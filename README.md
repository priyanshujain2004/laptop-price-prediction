# Laptop Price Prediction App

This project is a **Laptop Price Prediction Web App** built using **Streamlit**, **Scikit-Learn**, and **Pickle**. The app allows users to input various laptop specifications and predicts the estimated price using a pre-trained machine learning model.

---

## **Features**
- Predicts laptop prices based on:
  - Company
  - Type
  - Screen Size
  - CPU
  - RAM
  - GPU
  - Operating System
  - Weight
  - Screen Resolution (Width and Height)
  - Presence of SSD and HDD storage

- Uses a **Random Forest Regressor** trained on a real-world dataset.
- The model is saved using `pickle` to improve app performance by avoiding retraining.

---

## **Project Structure**

```plaintext
laptop-price-prediction/
│
├── model.py                # Model training and saving logic
├── app.py                  # Streamlit app interface
├── laptop_data.csv         # Dataset for training
├── model.pkl               # Saved pre-trained model
├── requirements.txt        # Dependencies
└── README.md               # Project description (this file)
```

---

## **Technologies Used**

- **Python**
- **Streamlit**: For building the web interface
- **Scikit-Learn**: For data preprocessing and model training
- **Pandas**: For data manipulation
- **Pickle**: For saving and loading the trained model

---

## **How to Run Locally**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/laptop-price-prediction.git
   cd laptop-price-prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train and Save the Model** (one-time step):
   ```bash
   python -c "from model import load_data, train_and_save_model; train_and_save_model(load_data())"
   ```

   This will create a `model.pkl` file containing the pre-trained model.

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

5. Open `http://localhost:8501` in your browser to access the app.

---

## **Deployment on Streamlit Cloud**

1. **Create a GitHub Repository** and upload all project files.
2. **Go to [Streamlit Cloud](https://share.streamlit.io/)** and log in with your GitHub account.
3. **Deploy the App**:
   - Select the GitHub repository.
   - Set `app.py` as the entry point.
   - Click **Deploy**.
4. Share the provided URL to access the app online.

---

## **Model Training**

The model uses the following steps:

1. **Data Cleaning**:
   - Removes unit labels (e.g., "GB", "kg") from features.
   - Extracts screen width and height from resolution.
   - Identifies the presence of SSD and HDD.
   - Caps outliers in the price column to a maximum of ₹500,000.

2. **Model**:
   - **Random Forest Regressor** with hyperparameters tuned for better performance.
   - Log transformation is applied to the price to stabilize variance.

3. **Saving the Model**:
   - The trained model is saved to `model.pkl` using `pickle`.

---

## **Future Improvements**
- Hyperparameter tuning using Grid Search or Randomized Search.
- Explore other regression algorithms like **XGBoost** or **LightGBM**.
- Add more features such as battery life or display type.

---

## **License**
This project is open-source and available under the [MIT License](LICENSE).

---

## **Acknowledgments**
- Special thanks to the creators of **Streamlit** and **Scikit-Learn**.

For questions or feedback, feel free to contact me!

