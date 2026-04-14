import os
import re
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from mysql.connector import pooling

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# DB Config
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
        pool_size=5,
        pool_reset_session=True,
        **db_config
    )
    print("Successfully connected to Aiven MySQL")
except Exception as e:
    print(f"Database Connection Error: {e}")

def get_db_connection():
    return connection_pool.get_connection()

# AI Models Setup
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def classifier(text):
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=10)
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return [{"label": result[0][0]['label']}]
        return [{"label": "neutral"}]
    except:
        return [{"label": "neutral"}]

client = Groq(api_key=os.environ.get("API_KEY"))

def clean_ai_text(text):
    text = re.sub(r':[a-z_]+:', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'^(AI|Response|Text|Face|Mentor|Vibe|Harish):\s*', '', text, flags=re.IGNORECASE)
    return ' '.join(text.split()).strip()

@app.route('/analyze', methods=['POST'])
def analyze():
    # Ping / Pre-warm Handler
    if request.is_json:
        ping_data = request.get_json()
        if ping_data.get('ping'):
            return jsonify({"status": "Backend & DB Warm"}), 200

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    try:
        user_text = ""
        face_mood = "Neutral"
        input_source = "Text"
        session_id = "global" # Default

        # 1. DUAL-INPUT & SESSION HANDLING
        if request.is_json:
            data = request.json
            user_text = data.get('text', '').strip()
            face_mood = data.get('face', 'Neutral')
            session_id = data.get('session_id', 'global')
        else:
            face_mood = request.form.get('face', 'Neutral')
            session_id = request.form.get('session_id', 'global')
            input_source = "Voice"
            if 'audio' in request.files:
                audio_file = request.files['audio']
                temp_path = "temp_voice.webm"
                audio_file.save(temp_path)
                with open(temp_path, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        file=(temp_path, file.read()),
                        model="whisper-large-v3", 
                        response_format="text",
                        language="en"
                    )
                user_text = transcription.strip()
                os.remove(temp_path)

        if not user_text and face_mood == "Neutral":
            return jsonify({"advice": "I'm listening. Whenever you're ready."})

        # 2. EMOTION ANALYSIS
        text_sentiment = "neutral"
        if user_text:
            text_analysis = classifier(user_text.lower())[0]
            text_sentiment = text_analysis['label'] 

        # 3. SESSION-SPECIFIC HISTORY RECALL
        # Get last 3 vibes for THIS device to check for spiral
        cursor.execute("SELECT is_positive FROM vibe_history WHERE session_id = %s ORDER BY timestamp DESC LIMIT 3", (session_id,))
        history_vibes = cursor.fetchall()
        is_spiral = len(history_vibes) >= 3 and all(not h['is_positive'] for h in history_vibes)
        
        recall_win = ""
        if is_spiral:
            # Find a positive moment for THIS device
            cursor.execute("SELECT user_text FROM vibe_history WHERE session_id = %s AND is_positive = 1 ORDER BY RAND() LIMIT 1", (session_id,))
            past_happy = cursor.fetchone()
            if past_happy:
                recall_win = f"They are in a loop. Recall their strength from when they said: '{past_happy['user_text']}'"

        # 4. CONTEXTUAL PROMPT (Filtered by Session ID)
        cursor.execute("SELECT user_text, ai_response FROM vibe_history WHERE session_id = %s ORDER BY timestamp DESC LIMIT 2", (session_id,))
        prev = cursor.fetchall()
        chat_context = " | ".join([f"U: {c['user_text']} A: {c['ai_response']}" for c in reversed(prev)])

        system_prompt = (
            f"Role: High-EQ Mentor. Face={face_mood}, Tone={text_sentiment}. "
            f"Note: {recall_win} Context: {chat_context}. "
            "STRICT: Max 20 words. No metadata/emojis. Respond only in English. Stay in the Flow. Be deeply obervant and empathetic. No persistant storage of name unless explictly told"
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text if user_text else "..."}]
        )
        
        advice = clean_ai_text(response.choices[0].message.content)

        # 5. SAVE & COMMIT WITH SESSION ID
        is_pos = text_sentiment in ['joy', 'love', 'surprise'] and face_mood != "Sad"
        cursor.execute("""
            INSERT INTO vibe_history (timestamp, user_text, ai_response, mood_detected, is_positive, session_id) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (datetime.now(), user_text, advice, f"{face_mood}/{text_sentiment}", is_pos, session_id))
        
        db.commit()
        return jsonify({"advice": advice})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"advice": "I'm right here with you."}), 500
    finally:
        cursor.close()
        db.close()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
