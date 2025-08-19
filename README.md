# Predictive-Analytics-for-Stock-Market-Using-LSTM


# Google Stock Price Prediction using LSTM

![GitHub language count](https://img.shields.io/github/languages/count/your-username/your-repo-name)
![GitHub top language](https://img.shields.io/github/languages/top/your-username/your-repo-name)
![GitHub last commit](https://img.shields.io/github/last-commit/your-username/your-repo-name)

## Overview

This project aims to predict the future stock prices of Google (GOOG) using a Long Short-Term Memory (LSTM) neural network. LSTMs are a type of recurrent neural network (RNN) well-suited for time-series data, like stock prices, because they can remember patterns over long periods.

The model is built using Keras with a TensorFlow backend. It is trained on historical stock data from 2014 to 2022 and then tested on data from 2022 to 2024 to evaluate its predictive performance.

---

## 📈 Features

-   **Historical Data**: Fetches over a decade of Google's stock data using the `yfinance` library.
-   **Data Preprocessing**: Includes data cleaning, normalization using `MinMaxScaler`, and preparing the data for the time-series model.
-   **LSTM Model**: Implements a stacked LSTM network with Dropout layers to prevent overfitting.
-   **Data Visualization**: Uses `matplotlib` to create insightful visualizations, including moving averages and a comparison of predicted vs. actual stock prices.

---

## 🛠️ Technologies Used

-   **Python**: The core programming language.
-   **Pandas**: For data manipulation and analysis.
-   **NumPy**: For numerical operations.
-   **yfinance**: To download historical stock market data from Yahoo Finance.
-   **Scikit-learn**: For data preprocessing, specifically `MinMaxScaler`.
-   **Keras (with TensorFlow backend)**: For building and training the LSTM model.
-   **Matplotlib**: For creating visualizations.
-   **Jupyter Notebook**: For interactive development and presentation.

---

## ⚙️ Installation

To run this project, you'll need to have Python and the required libraries installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the necessary libraries:**
    ```bash
    pip install numpy pandas yfinance matplotlib scikit-learn tensorflow
    ```

---

## 🚀 Usage

You can run the project by executing the `stock_price_pred.ipynb` file in a Jupyter Notebook environment.

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  **Open and run the notebook:**
    Open the `stock_price_pred.ipynb` file and run the cells sequentially to see the data processing, model training, and final predictions.

---

## 📊 Results and Visuals

### Moving Averages

The 100-day moving average is plotted against the closing price to visualize the stock's trend over time.
<img width="676" height="505" alt="image" src="https://github.com/user-attachments/assets/4c5b06d2-93ba-4abd-b558-6908db82d165" />

### Prediction Results

The final graph shows a comparison between the original stock prices and the prices predicted by the LSTM model. The model successfully captures the general trend of the stock price, indicating its effectiveness.

<img width="850" height="679" alt="image" src="https://github.com/user-attachments/assets/34015782-e6b4-4b7d-9c2b-c9a304566363" />


---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
