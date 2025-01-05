# Next Word Prediction using LSTM and GRU

## Overview
This project focuses on developing a deep learning-based next-word prediction model using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks. The model leverages the complex text of Shakespeare's *Hamlet* from the NLTK Gutenberg dataset to predict the next word in a given sequence. A Streamlit web application is deployed for real-time interaction with the model.

## Project Features
- **LSTM-based Model:** Achieved 65% accuracy on training for 100 epochs.
- **GRU-based Model:** Provides an alternative architecture for comparison and experimentation.
- **Early Stopping:** Implemented to monitor validation loss and prevent overfitting.
- **Web Deployment:** User-friendly Streamlit app for real-time word predictions.

## File Structure
```
next-word-prediction-project/
├── Early_Stopping_GRU.h5        # Trained GRU model with early stopping
├── Early_Stopping_LSTM.h5       # Trained LSTM model with early stopping
├── HamletData.txt               # Text data from Shakespeare's *Hamlet*
├── LSTM.h5                      # Trained LSTM model without early stopping
├── NextWordPrediction.ipynb     # Jupyter Notebook for training and analysis
├── app.py                       # Streamlit app for deployment
├── requirements.txt             # Dependencies for the project
└── tokenizer.pickle             # Tokenizer object for text preprocessing
```

## Key Steps

### 1. Data Collection
The dataset used is the text of *Hamlet* by Shakespeare, sourced from the NLTK Gutenberg dataset. This text provides rich and complex patterns, making it an excellent choice for training a sequence prediction model.

### 2. Data Preprocessing
- Tokenized the text into sequences using NLTK.
- Padded sequences to ensure uniform input lengths.
- Split the data into training and testing sets.

### 3. Model Building
- **LSTM Model Architecture:**
  - Embedding layer for word representation.
  - Two LSTM layers for sequence modeling.
  - Dense output layer with softmax activation for next-word probability prediction.
- **GRU Model:** Built with a similar structure to explore its performance compared to LSTM.

### 4. Model Training
- Trained using prepared sequences with early stopping to monitor validation loss.
- Achieved 65% accuracy on the LSTM model after 100 epochs.

### 5. Model Evaluation
The model’s performance is evaluated using example sentences. It predicts the next word based on the learned patterns and context from the dataset.

### 6. Deployment
A Streamlit app provides an interactive interface where users can input a sequence of words and get the model’s next-word prediction in real time.

### Deployment Link
Access the deployed application at: [Next Word Predictor](https://nextwordpredictor-fftadagpfxygmmgtvhset4.streamlit.app/)

## Requirements
To set up the project locally, install the dependencies listed in `requirements.txt`:
```
pip install -r requirements.txt
```

## How to Use
1. Clone the repository and navigate to the project directory.
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Access the app through the provided local URL.
4. Input a sequence of words in the text box to get the predicted next word.

## Future Enhancements
- Experiment with different datasets to test the model’s adaptability.
- Implement bidirectional LSTM/GRU for improved context understanding.
- Extend the app’s functionality to handle multi-word predictions.

## Acknowledgments
- *Hamlet* text sourced from the NLTK Gutenberg dataset.
- Developed with TensorFlow and Keras for deep learning.
- Streamlit for web deployment.

---

Feel free to explore and enhance this project!
![Uploading Screenshot 2025-01-05 at 7.58.46 PM.png…]()

