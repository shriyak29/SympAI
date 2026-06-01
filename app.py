import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from rapidfuzz import process, fuzz
from openai import OpenAI

# ─────────────────────────────────────────
# Page Configuration — must be first
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SympAI — Disease Predictor",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────
# Load API key from Streamlit secrets
# ─────────────────────────────────────────
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    api_key = os.getenv("GOOGLE_API_KEY", "")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
) if api_key else None

# ─────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model/disease_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

@st.cache_data
def load_csv():
    desc_df = pd.read_csv('data/symptom_Description.csv')
    prec_df = pd.read_csv('data/symptom_precaution.csv')
    desc_df.columns = desc_df.columns.str.strip()
    prec_df.columns = prec_df.columns.str.strip()
    return desc_df, prec_df

data          = load_model()
model         = data['model']
le            = data['label_encoder']
all_symptoms  = data['all_symptoms']
severity_dict = data['severity_dict']
desc_df, prec_df = load_csv()

# ─────────────────────────────────────────
# Alias Map
# ─────────────────────────────────────────
ALIASES = {
    "fever"                 : ["high_fever","mild_fever"],
    "high fever"            : ["high_fever"],
    "mild fever"            : ["mild_fever"],
    "low grade fever"       : ["mild_fever"],
    "feverish"              : ["mild_fever"],
    "feel hot"              : ["mild_fever"],
    "temperature"           : ["mild_fever"],
    "cough"                 : ["cough"],
    "coughing"              : ["cough"],
    "dry cough"             : ["cough"],
    "wet cough"             : ["cough","phlegm"],
    "phlegm"                : ["phlegm"],
    "cold"                  : ["runny_nose","chills"],
    "flu"                   : ["high_fever","chills","cough","muscle_pain","headache"],
    "common cold"           : ["runny_nose","chills","cough","continuous_sneezing"],
    "runny nose"            : ["runny_nose"],
    "blocked nose"          : ["congestion"],
    "sneezing"              : ["continuous_sneezing"],
    "sore throat"           : ["throat_irritation"],
    "throat pain"           : ["throat_irritation"],
    "headache"              : ["headache"],
    "head pain"             : ["headache"],
    "migraine"              : ["headache"],
    "body ache"             : ["muscle_pain","joint_pain"],
    "body pain"             : ["muscle_pain","joint_pain"],
    "whole body hurts"      : ["muscle_pain","joint_pain","back_pain"],
    "pain everywhere"       : ["muscle_pain","joint_pain","back_pain"],
    "muscle pain"           : ["muscle_pain"],
    "joint pain"            : ["joint_pain"],
    "back pain"             : ["back_pain"],
    "neck pain"             : ["neck_pain"],
    "stiff neck"            : ["stiff_neck"],
    "chest pain"            : ["chest_pain"],
    "stomach pain"          : ["stomach_pain"],
    "stomach ache"          : ["stomach_pain"],
    "tummy ache"            : ["stomach_pain"],
    "numbness"              : ["numbness_tingling"],
    "numb"                  : ["numbness_tingling"],
    "tingling"              : ["numbness_tingling"],
    "fatigue"               : ["fatigue"],
    "tired"                 : ["fatigue"],
    "exhausted"             : ["fatigue"],
    "no energy"             : ["fatigue","weakness"],
    "weakness"              : ["weakness"],
    "weak"                  : ["weakness"],
    "feel weak"             : ["weakness"],
    "lethargy"              : ["lethargy"],
    "lethargic"             : ["lethargy"],
    "drowsy"                : ["lethargy"],
    "dizziness"             : ["dizziness"],
    "dizzy"                 : ["dizziness"],
    "lightheaded"           : ["dizziness"],
    "vertigo"               : ["dizziness"],
    "nausea"                : ["nausea"],
    "nauseous"              : ["nausea"],
    "feel sick"             : ["nausea"],
    "feel like vomiting"    : ["nausea","vomiting"],
    "throwing up"           : ["vomiting"],
    "vomiting"              : ["vomiting"],
    "vomit"                 : ["vomiting"],
    "puking"                : ["vomiting"],
    "diarrhea"              : ["diarrhoea"],
    "diarrhoea"             : ["diarrhoea"],
    "loose stools"          : ["diarrhoea"],
    "loose motions"         : ["diarrhoea"],
    "loose motion"          : ["diarrhoea"],
    "constipation"          : ["constipation"],
    "bloating"              : ["distention_of_abdomen"],
    "indigestion"           : ["indigestion"],
    "acidity"               : ["acidity"],
    "heartburn"             : ["acidity"],
    "no appetite"           : ["loss_of_appetite"],
    "loss of appetite"      : ["loss_of_appetite"],
    "breathlessness"        : ["breathlessness"],
    "shortness of breath"   : ["breathlessness"],
    "cant breathe"          : ["breathlessness"],
    "difficulty breathing"  : ["breathlessness"],
    "chest tightness"       : ["chest_pain","breathlessness"],
    "palpitations"          : ["palpitations"],
    "heart racing"          : ["palpitations"],
    "red eyes"              : ["redness_of_eyes"],
    "watery eyes"           : ["watering_from_eyes"],
    "blurred vision"        : ["blurred_and_distorted_vision"],
    "yellow eyes"           : ["yellowing_of_skin"],
    "rash"                  : ["skin_rash"],
    "skin rash"             : ["skin_rash"],
    "itching"               : ["itching"],
    "itchy"                 : ["itching"],
    "yellow skin"           : ["yellowing_of_skin"],
    "jaundice"              : ["yellowing_of_skin","yellow_urine"],
    "pale skin"             : ["pale_complexion"],
    "swelling"              : ["swelled_lymph_nodes"],
    "confused"              : ["mental_confusion"],
    "confusion"             : ["mental_confusion"],
    "feel confused"         : ["mental_confusion"],
    "brain fog"             : ["mental_confusion"],
    "anxiety"               : ["anxiety"],
    "depression"            : ["depression"],
    "mood swings"           : ["mood_swings"],
    "chills"                : ["chills"],
    "shivering"             : ["shivering","chills"],
    "sweating"              : ["sweating"],
    "night sweats"          : ["sweating"],
    "burning urination"     : ["burning_micturition"],
    "frequent urination"    : ["polyuria"],
    "dark urine"            : ["dark_urine"],
    "weight loss"           : ["weight_loss"],
    "weight gain"           : ["weight_gain"],
    "hair loss"             : ["hair_loss"],
    "thirsty"               : ["excessive_hunger"],
    "excessive thirst"      : ["excessive_hunger"],
    "dry mouth"             : ["dryness_and_tingling_lips"],
    "cold hands"            : ["cold_hands_and_feets"],
    "cold feet"             : ["cold_hands_and_feets"],
    "coma"                  : ["altered_sensorium"],
    "unconscious"           : ["altered_sensorium"],
    "slurred speech"        : ["slurred_speech"],
    "coughing blood"        : ["blood_in_sputum"],
}

SEVERE = [
    "chest pain", "cant breathe", "breathlessness",
    "shortness of breath", "difficulty breathing",
    "unconscious", "coma", "coughing blood",
    "blood in vomit", "slurred speech",
    "heart racing", "palpitations",
]

MIN_SYMPTOMS = 3
PREDICTION_STAGES = {
    3: {'show': 1, 'min_conf': 20},
    5: {'show': 2, 'min_conf': 15},
    7: {'show': 3, 'min_conf': 10},
}

def get_stage(count):
    if count >= 7:
        return PREDICTION_STAGES[7]
    elif count >= 5:
        return PREDICTION_STAGES[5]
    else:
        return PREDICTION_STAGES[3]

# ─────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────
def match_symptoms(user_input):
    user_lower = user_input.lower().strip()
    found      = set()

    fillers = [
        "i have been having","i have been feeling","i have been",
        "i am having","i am feeling","i am experiencing",
        "i feel like i have","i feel like","i've been",
        "i'm having","i'm feeling","i got","i get",
        "i have","i feel","i am","i've","i'm",
        "suffering from","experiencing","having","getting",
        "since yesterday","since today","for a while",
        "really","very","quite","slightly","a bit of",
        "also","too","and","with","plus",
    ]
    cleaned = user_lower
    for f in sorted(fillers, key=len, reverse=True):
        cleaned = cleaned.replace(f, ' ')
    cleaned = ' '.join(cleaned.split())

    for phrase, mapped in sorted(
        ALIASES.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if phrase in user_lower or phrase in cleaned:
            for s in mapped:
                if s in all_symptoms:
                    found.add(s)

    for symptom in all_symptoms:
        readable = symptom.replace('_', ' ')
        if readable in user_lower or readable in cleaned:
            found.add(symptom)

    words = [w for w in cleaned.replace(',', ' ').split() if len(w) > 4]
    for symptom in all_symptoms:
        for sw in symptom.replace('_', ' ').split():
            if len(sw) > 4:
                for w in words:
                    if w == sw or (len(w) > 5 and len(sw) > 5 and (w in sw or sw in w)):
                        found.add(symptom)

    readable_map = {s.replace('_', ' '): s for s in all_symptoms}
    all_words    = cleaned.split()
    chunks       = set()
    for i in range(len(all_words) - 1):
        chunks.add(f"{all_words[i]} {all_words[i+1]}")

    for chunk in chunks:
        match = process.extractOne(chunk, readable_map.keys(), scorer=fuzz.ratio)
        if match and match[1] >= 90:
            found.add(readable_map[match[0]])

    if 'cold hands' not in user_lower and 'cold feet' not in user_lower:
        found.discard('cold_hands_and_feets')
    if 'chest' not in user_lower:
        found.discard('chest_pain')
    if 'eye' not in user_lower and 'vision' not in user_lower:
        found.discard('redness_of_eyes')
        found.discard('blurred_and_distorted_vision')
    if 'neck' not in user_lower:
        found.discard('stiff_neck')
        found.discard('neck_pain')
    if 'hair' not in user_lower:
        found.discard('hair_loss')
    if 'weight' not in user_lower:
        found.discard('weight_loss')
        found.discard('weight_gain')

    return list(found)

def predict_disease(symptoms_list):
    vec   = [1 if s in symptoms_list else 0 for s in all_symptoms]
    df_in = pd.DataFrame([vec], columns=all_symptoms)
    probs = model.predict_proba(df_in)[0]

    stage    = get_stage(len(symptoms_list))
    max_show = stage['show']
    min_conf = stage['min_conf']

    top = np.argsort(probs)[-10:][::-1]
    results = []
    for idx in top:
        disease = le.inverse_transform([idx])[0]
        conf    = round(probs[idx] * 100, 1)
        if conf >= min_conf:
            results.append((disease, conf))

    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        idx     = np.argmax(probs)
        disease = le.inverse_transform([idx])[0]
        conf    = round(probs[idx] * 100, 1)
        return [(disease, conf)]

    return results[:max_show]

def get_description(disease):
    row = desc_df[desc_df['Disease'].str.lower() == disease.lower()]
    if not row.empty:
        return row.iloc[0]['Description']
    return "No description available."

def get_precautions(disease):
    row = prec_df[prec_df['Disease'].str.lower() == disease.lower()]
    if not row.empty:
        precs = [row.iloc[0][f'Precaution_{i}'] for i in range(1, 5)]
        return [p for p in precs if pd.notna(p)]
    return ["Consult a doctor immediately."]

def check_severity(user_input):
    u = user_input.lower()
    return [s for s in SEVERE if s in u]

# ─────────────────────────────────────────
# UI — Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/3/3b/BMS_College_of_Engineering_logo.png", width=100)
    st.title("🏥 SympAI")
    st.caption("AI Disease Predictor Chatbot")
    st.divider()

    st.subheader("📊 Session Info")
    symptom_count = len(st.session_state.get("session_symptoms", []))
    st.metric("Symptoms Noted", symptom_count)

    if st.session_state.get("session_symptoms"):
        st.subheader("🔍 Symptoms So Far")
        for s in st.session_state.session_symptoms:
            st.write(f"• {s.replace('_', ' ')}")

    st.divider()

    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.session_symptoms = []
        st.session_state.messages         = []
        st.rerun()

    st.divider()
    st.caption("⚕️ For informational purposes only. Always consult a real doctor.")

# ─────────────────────────────────────────
# UI — Main Chat
# ─────────────────────────────────────────
st.title("🏥 SympAI: Disease Predictor Chatbot")
st.caption("Describe your symptoms naturally — I understand plain language!")

# Initialise session state
if "session_symptoms" not in st.session_state:
    st.session_state.session_symptoms = []
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role"   : "assistant",
        "content": "👋 Hello! I'm SympAI, your AI Disease Predictor.\n\nDescribe your symptoms naturally and I'll identify possible conditions with precautions.\n\n💡 Example: *'I have fever, headache and feel very weak'*\n\nType **'clear'** to reset the session."
    }]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Describe your symptoms here..."):

    # Handle clear command
    if prompt.strip().lower() == "clear":
        st.session_state.session_symptoms = []
        st.session_state.messages         = []
        st.rerun()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # Severity check
        warnings = check_severity(prompt)
        if warnings:
            st.error(f"🚨 **WARNING:** {', '.join(warnings)} detected! Please seek **immediate medical attention.**")

        # Match symptoms
        found = match_symptoms(prompt)
        new   = [s for s in found if s not in st.session_state.session_symptoms]
        st.session_state.session_symptoms.extend(new)

        count = len(st.session_state.session_symptoms)

        if not found:
            response = "❌ I couldn't identify any symptoms in that.\n\nTry describing physical feelings like:\n*'I have fever, feel dizzy and very tired'*"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        elif count < MIN_SYMPTOMS:
            needed   = MIN_SYMPTOMS - count
            readable = [s.replace('_', ' ') for s in new]
            response = f"✅ Noted: **{', '.join(readable)}**\n\nPlease describe **{needed} more symptom(s)** to begin prediction."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            if new:
                readable = [s.replace('_', ' ') for s in new]
                st.markdown(f"✅ Noted: **{', '.join(readable)}**")

            predictions = predict_disease(st.session_state.session_symptoms)

            if count < 5:
                st.markdown("### 🎯 Most Likely Condition:")
            elif count < 7:
                st.markdown("### 🎯 Most Likely + Possible Conditions:")
            else:
                st.markdown("### 🎯 Full Assessment:")

            full_response = ""
            for rank, (disease, conf) in enumerate(predictions, 1):
                label = "✅ MOST LIKELY" if rank == 1 else "🔸 ALSO POSSIBLE"
                desc  = get_description(disease)
                precs = get_precautions(disease)

                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{label}: {disease}**")
                    with col2:
                        st.markdown(f"**{conf}%**")

                    st.progress(conf / 100)
                    st.info(f"ℹ️ {desc}")

                    prec_text = " | ".join(precs)
                    st.warning(f"⚠️ **Precautions:** {prec_text}")
                    st.divider()

                full_response += f"{label}: {disease} ({conf}%)\n"

            if count < 7:
                st.markdown("💡 *Add more symptoms to refine the prediction.*")

            st.error("⚕️ These are **possible conditions only**. Please consult a real doctor for proper diagnosis.")
            st.session_state.messages.append({"role": "assistant", "content": full_response})