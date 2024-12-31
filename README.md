# Sentiment Analysis Streamlit App

This repository contains a Streamlit application for sentiment analysis using a Word2Vec model. The model was trained on the Kindle review dataset and achieved a test accuracy of 77%.

## Live Demo
Check out the live demo [here](https://sentiment-analysis-kindle-dataset.streamlit.app/).

## Installation

To run the application locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Himank-Khatri/Kindle-Sentiment-Analysis.git
   cd Kindle-Sentiment-Analysis
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## Model Training

The model training and experiments are documented in `experiments.py`. Achieved accuracy of 77% using random forest classifier. The trained model and Word2Vec vectors are saved in the `artifacts` directory.

## Files and Directories

- **app.py**: The main Streamlit application file.
- **experiments.py**: Contains the code for model training and experiments.
- **artifacts/**: Directory containing the trained model and Word2Vec vectors.
  - `classifier.pkl`: The trained classifier model.
  - `wv_model.pkl`: The Word2Vec model.
- **requirements.txt**: List of required Python packages.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

