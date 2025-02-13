# Laptop Price Prediction App

This project is a **Laptop Price Prediction Web App** built using **Streamlit** and **Scikit-Learn**. The app allows users to input various laptop specifications and predicts the estimated price using a machine learning model.

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

---

## **Project Structure**

```plaintext
laptop-price-prediction/
│
├── model.py                # Model training and feature engineering logic
├── app.py                  # Streamlit app interface
├── laptop_data.csv         # Dataset for training
├── requirements.txt        # Dependencies
└── README.md               # Project description (this file)
```

---

## **Technologies Used**

- **Python**
- **Streamlit**: For building the web interface
- **Scikit-Learn**: For data preprocessing and model training
- **Pandas**: For data manipulation

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

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

4. Open `http://localhost:8501` in your browser to access the app.

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

3. **Evaluation**:
   - Mean Squared Error (MSE) is used to evaluate the model’s accuracy.
   - Cross-validation ensures generalization and avoids overfitting.

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

