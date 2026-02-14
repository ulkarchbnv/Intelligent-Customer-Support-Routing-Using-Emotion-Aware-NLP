# 🤖 Intelligent Customer Support Routing Chatbot

This repository contains a local Streamlit web application that uses an Emotion-Aware Natural Language Processing (NLP) model to automatically triage and route customer support tickets. 

By combining a trained Machine Learning model (`SGDClassifier` + `TF-IDF`) with a Hybrid Business Rules layer, the bot categorizes customer messages into five actionable priority levels (Angry, Frustrated, Confused, Satisfied, Calm) and routes them to the appropriate team.

## ✨ Features
* **Real-time UI:** A chat-based interface built with Streamlit.
* **Hybrid Routing:** Catches obvious high-risk triggers (like ALL CAPS shouting or specific keywords) instantly via business rules, while utilizing the NLP model for nuanced emotional text.
* **Auto-Training:** The app automatically downloads the `go_emotions` dataset and trains the ML pipeline upon the first boot, caching it for lightning-fast performance afterward.

---

## 🚀 How to Run Locally

### Prerequisites
Make sure you have **Python 3.8 or higher** installed on your computer.

Open your terminal or command prompt and run:
```bash
git clone [https://github.com/ulkarchbnv/Intelligent-Customer-Support-Routing-Using-Emotion-Aware-NLP.git](https://github.com/ulkarchbnv/Intelligent-Customer-Support-Routing-Using-Emotion-Aware-NLP.git)
cd Intelligent-Customer-Support-Routing-Using-Emotion-Aware-NLP
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

```

## 📊 Read the Full Data Report
To dive deeper into the methodology, exploratory data analysis, and business conclusions, please view the full report here: 
👉 **[Data Report.pdf](https://github.com/ulkarchbnv/Intelligent-Customer-Support-Routing-Using-Emotion-Aware-NLP/blob/main/Data%20Report.pdf)**

## 🎥 Watch the Demo
See the Intelligent Support Router in action! Click the thumbnail below to watch a quick demonstration of the chatbot handling different customer priorities:

[![Watch the Demo Video](https://img.youtube.com/vi/TWV79WouL6o/maxresdefault.jpg)](https://youtu.be/TWV79WouL6o)







