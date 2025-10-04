# 🎯 Customer Churn Predictor using Artificial Neural Networks
A machine learning application that predicts whether a bank customer is likely to leave (churn) based on their profile and behavior. Built with TensorFlow and deployed using Streamlit for an interactive user experience.
!("C:\Users\piyus\Pictures\Saved Pictures\customer-churn.jpg")

## 🌟 What Does This App Do?
This application helps banks identify customers who might leave their services. By analyzing customer data like age, balance, credit score, and activity patterns, the AI model predicts the likelihood of customer churn. This allows banks to:

 - 🎯 **Proactively retain customers** by identifying at-risk individuals
 - 💡 **Make data-driven decisions** about customer retention strategies
 - 📊 **Understand patterns** that lead to customer dissatisfaction
 - 💰 **Save costs** by preventing customer loss

## 🛠️ Technologies Used

 - **Python 3.11** - Programming language
 - **TensorFlow/Keras** - Deep learning framework for building the neural network
 - **Pandas** - Data manipulation and analysis
 - **NumPy** - Numerical computations
 - **Scikit-learn** - Data preprocessing and model evaluation
 - **Streamlit** - Web application framework
 - **Pickle** - Model serialization

## 📊 How to Use
- **Step 1: Clone the Repository**
 ```
git clone https://github.com/Piyush-Yadav2506/CHURN_PREDICTOR_ANN.git
cd CHURN_PREDICTOR_ANN
```
- **Step 2: Install Required Packages**
 ```pip install -r requirements.txt```
- **Step 3: Run the Application**
 ```streamlit run app.py```
 The app will open automatically in your default web browser at ```http://localhost:8501```

- **Step 4:**
**Enter Customer Details** - Fill in the form with customer information:

  * Personal info (Age, Gender, Geography)
  * Financial details (Credit Score, Balance, Salary)
  * Account info (Tenure, Products, Credit Card status)


**Click Predict** - Press the "🚀 Predict Churn Risk" button
**View Results** - Get instant predictions with:
 - Churn probability percentage
 - Risk level classification
 - Model confidence score
 - Detailed insights and recommendations

## 🧠 Model Architecture
The prediction model uses an Artificial Neural Network (ANN) with the following structure:

 - Input Layer: Customer features (10 parameters)
 - Hidden Layers: Multiple dense layers with activation functions
 - Output Layer: Single neuron with sigmoid activation (probability output)
 - Optimizer: Adam
 - Loss Function: Binary Cross-entropy

## 📈 Model Performance

 - Accuracy: ~87% 
 - Precision: ~82%
 - Recall: ~86%
 - F1-Score: ~84%

## 🎨 Screenshots
 ### Main Dashboard
 
