import os
import re
import uuid
import requests
import librosa
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from mysql.connector import pooling

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# 1. DATABASE CONFIGURATION
db_config = {
    "host": os.environ.get("DB_HOST", "mysql-274b11d4-harishjhr05-a436.e.aivencloud.com"),
    "user": os.environ.get("DB_USER", "avnadmin"), 
    "password": os.environ.get("DB_PASSWORD"),
    "database": os.environ.get("DB_NAME", "defaultdb"),
    "port": int(os.environ.get("DB_PORT", 15296))
}

try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="emotepool",
        pool_size=10,
        pool_reset_session=True,
        **db_config
    )
except Exception as e:
    print(f"Database Connection Error: {e}")
    connection_pool = None

def get_db_connection():
    return connection_pool.get_connection()

# 2. AI & SIGNAL PROCESSING UTILITIES
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
client = Groq(api_key=os.environ.get("API_KEY"))

def get_text_sentiment(text):
    if not text: return "neutral"
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=5)
        result = response.json()
        return result[0][0]['label'].lower() if isinstance(result, list) else "neutral"
    except:
        return "neutral"

def analyze_audio_vibe(file_path):
    try:
        y, sr = librosa.load(file_path, duration=10)
        energy = np.mean(librosa.feature.rms(y=y))
        if energy < 0.008: return "Whispering/Defeated"
        if energy < 0.02: return "Low/Somber"
        if energy > 0.08: return "High/Intense"
        return "Stable/Normal"
    except:
        return "Indeterminate"

def clean_ai_text(text):
    text = re.sub(r':[a-z_]+:', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'^(AI|Friend|Response|Mentor):\s*', '', text, flags=re.IGNORECASE)
    return ' '.join(text.split()).strip()

# 3. MAIN AFFECTIVE ENGINE
@app.route('/analyze', methods=['POST'])
def analyze():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    # Unique filename prevents crashes during multi-user access or quick testing
    temp_path = f"voice_{uuid.uuid4()}.webm"
    
    try:
        user_text, face_mood, voice_vibe = "", "Not Detected", "No Audio"
        
        # A. INPUT HANDLING
        if request.is_json:
            data = request.json
            user_text = data.get('text', '').strip()
            face_mood = data.get('face', 'Neutral')
        else:
            face_mood = request.form.get('face', 'Neutral')
            if 'audio' in request.files:
                audio_file = request.files['audio']
                audio_file.save(temp_path)
                voice_vibe = analyze_audio_vibe(temp_path)
                with open(temp_path, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        file=(temp_path, file.read()),
                        model="whisper-large-v3",
                        language="en"
                    )
                    # Corrected to .text for Groq API compliance
                    user_text = transcription.text.strip()
                if os.path.exists(temp_path): os.remove(temp_path)

        if not user_text and face_mood in ["Neutral", "Not Detected"]:
            return jsonify({"advice": "I'm listening. Whenever you're ready to share."})

        # B. THE EXACT INTEGRATED AFFECTIVE LOGIC MATRIX
        text_sentiment = get_text_sentiment(user_text)
        neg_ems = ['sadness', 'fear', 'anger']
        pos_ems = ['joy', 'love', 'surprise']
        masking_note = "Aligned"

        if face_mood == "Happy":
            if text_sentiment in neg_ems:
                masking_note = "The Brave Face: Smiling through pain/minimizing a problem."
            else:
                masking_note = "Pure Congruence: Genuine joy or success."
        
        elif face_mood == "Sad":
            if text_sentiment in pos_ems:
                masking_note = "The Faked Fine: Saying 'I'm okay' but looks visibly hurt or defeated."
            else:
                masking_note = "Total Distress: Overwhelmed and not trying to hide it."
        
        elif face_mood == "Frustrated":
            if text_sentiment in pos_ems:
                masking_note = "Sarcastic/Cynical: Saying something 'good' but looking annoyed."
            else:
                masking_note = "Vent Mode: High energy, aggressive or annoyed venting."
        
        elif face_mood == "Surprised":
            if text_sentiment in pos_ems:
                masking_note = "Excitement/Shock: Something unexpected but good has happened."
            else:
                masking_note = "Stunned/Disbelief: Reeling from bad news or a sudden setback."

        # C. HISTORY & SPIRAL DETECTION
        cursor.execute("SELECT is_positive FROM vibe_history ORDER BY timestamp DESC LIMIT 3")
        history = cursor.fetchall()
        is_spiral = len(history) >= 3 and all(h['is_positive'] == 0 for h in history)
        
        recall_win = ""
        if is_spiral:
            cursor.execute("SELECT user_text FROM vibe_history WHERE is_positive = 1 ORDER BY RAND() LIMIT 1")
            past = cursor.fetchone()
            if past: recall_win = f"Remind them of a better time: '{past['user_text']}'."

        # D. EXACT SYSTEM PROMPT (Friend Persona)
        cursor.execute("SELECT user_text, ai_response FROM vibe_history ORDER BY timestamp DESC LIMIT 2")
        chat_context = " | ".join([f"U: {c['user_text']} A: {c['ai_response']}" for c in reversed(cursor.fetchall())])

        system_prompt = (
            f"Role: High-EQ Virtual Friend. Current Context: {chat_context}. "
            f"Sensors: Face={face_mood}, Voice={voice_vibe}, Text={text_sentiment}. "
            f"Psychological Insight: {masking_note} {recall_win} "
            "STYLE: Professional but deeply warm. Do not circle back questions. "
            "If Face/Voice conflict with Text, prioritize the visual/audio cues. "
            "STRICT: Max 20 words. No emojis. Be a healing presence."
        )

        # E. GENERATE RESPONSE
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text if user_text else "..."}]
        )
        advice = clean_ai_text(response.choices[0].message.content)

        # F. DATABASE UPDATE
        is_pos = (text_sentiment in pos_ems) and (face_mood not in ["Sad", "Frustrated", "Angry"])
        cursor.execute(
            "INSERT INTO vibe_history (timestamp, user_text, ai_response, mood_detected, is_positive) VALUES (%s, %s, %s, %s, %s)",
            (datetime.now(), user_text, advice, f"F:{face_mood}/V:{voice_vibe}/T:{text_sentiment}", 1 if is_pos else 0)
        )
        db.commit()

        return jsonify({
            "advice": advice,
            "transcription": user_text,
            "analysis": {"face": face_mood, "voice": voice_vibe, "text": text_sentiment}
        })

    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return jsonify({"advice": "I'm right here with you."}), 500
    finally:
        cursor.close()
        db.close()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
