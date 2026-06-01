# 🏥 SympAI: Disease Predictor Chatbot

SympAI is an AI-powered chatbot that predicts possible diseases based on symptoms 
described in plain, natural language. Built entirely in Python, it runs fully 
offline — no internet connection needed for core prediction.


## 💬 How It Works

Describe your symptoms naturally — for example:
> *"I have fever, headache and feel very weak"*

SympAI extracts your symptoms, passes them to a trained machine learning model, 
and returns the most likely diseases with precautions.

Predictions strengthen as you describe more symptoms:
- 3 symptoms → Most likely disease
- 5 symptoms → Top 2 conditions
- 7+ symptoms → Full assessment with top 3


## ⚠️ Limitations

- Predicts **41 diseases only** based on the available Kaggle dataset
- Recognises **131 unique symptoms** — others may not be identified
- Does not account for age, gender, or medical history
- Not clinically validated — for health awareness only


## 🛠️ Tech Stack

- Python, scikit-learn, pandas, numpy
- Random Forest Classifier — **97%+ accuracy**
- RapidFuzz for symptom matching
- Google Gemini API via OpenAI SDK
- Streamlit for web interface

---

## 🚀 Run Locally
