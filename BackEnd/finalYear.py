import os
import re
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from mysql.connector import pooling

# Load .env only for local testing. 
# On Render/Aiven, these will be pulled from the dashboard environment settings.
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 1. DYNAMIC DATABASE CONNECTION (Updated for Aiven)
db_config = {
    "host": os.environ.get("DB_HOST", "mysql-274b11d4-harishjhr05-a436.e.aivencloud.com"),
    "user": os.environ.get("DB_USER", "avnadmin"), 
    "password": os.environ.get("DB_PASSWORD"), # Set this in Render Dashboard
    "database": os.environ.get("DB_NAME", "defaultdb"),
    "port": int(os.environ.get("DB_PORT", 15296))
}

# Use pooling to prevent "Lost Connection" errors
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

# 2. AI MODELS

HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def classifier(text):
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=10)
        result = response.json()
        
        # Hugging Face returns a list of lists: [[{'label': 'joy', 'score': 0.9}, ...]]
        # We just need the top label
        if isinstance(result, list) and len(result) > 0:
            return [{"label": result[0][0]['label']}]
        else:
            return [{"label": "neutral"}]
    except Exception as e:
        print(f"HF API Error: {e}")
        return [{"label": "neutral"}]
# classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# API Key handled via Environment Variable for security
client = Groq(api_key=os.environ.get("API_KEY"))

def clean_ai_text(text):
    text = re.sub(r':[a-z_]+:', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'^(AI|Response|Text|Face|Mentor|Vibe|Harish):\s*', '', text, flags=re.IGNORECASE)
    return ' '.join(text.split()).strip()

@app.route('/analyze', methods=['POST'])
def analyze():
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

        # 1. DUAL-INPUT HANDLING
        if request.is_json:
            data = request.json
            user_text = data.get('text', '').strip()
            face_mood = data.get('face', 'Neutral')
        else:
            face_mood = request.form.get('face', 'Neutral')
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
            return jsonify({"advice": "I'm listening. Whenever you're ready to share."})

        # 2. EMOTION ANALYSIS
        text_sentiment = "neutral"
        if user_text:
            text_analysis = classifier(user_text.lower())[0]
            text_sentiment = text_analysis['label'] 

        # 3. AFFECTIVE LOGIC ENGINE
        masking_note = ""
        negative_emotions = ['sadness', 'fear', 'anger']
        
        if face_mood == "Happy" and text_sentiment in negative_emotions:
            masking_note = "User is smiling but their words suggest pain. Acknowledge their bravery in masking."
        elif face_mood == "Neutral" and text_sentiment in negative_emotions:
            masking_note = "User is keeping a straight face but feeling low. Validate their internal strength."

        cursor.execute("SELECT is_positive FROM vibe_history ORDER BY timestamp DESC LIMIT 3")
        history_vibes = cursor.fetchall()
        is_spiral = len(history_vibes) >= 3 and all(not h['is_positive'] for h in history_vibes)
        
        recall_win = ""
        if is_spiral:
            cursor.execute("SELECT user_text FROM vibe_history WHERE is_positive = 1 ORDER BY RAND() LIMIT 1")
            past_happy = cursor.fetchone()
            if past_happy:
                recall_win = f"They are in a negative loop. Remind them of a time they felt better, like when they said: '{past_happy['user_text']}'"

        # 4. CONTEXTUAL PROMPT FUSION
        cursor.execute("SELECT user_text, ai_response FROM vibe_history ORDER BY timestamp DESC LIMIT 2")
        prev = cursor.fetchall()
        chat_context = " | ".join([f"U: {c['user_text']} A: {c['ai_response']}" for c in reversed(prev)])

        system_prompt = (
            f"Role: High-EQ Personal Mentor. Current Stats: Face={face_mood}, Tone={text_sentiment}, Source={input_source}. "
            f"Observation: {masking_note} {recall_win} Context: {chat_context}. "
            "STRICT: Max 20 words. No metadata/emojis. No gendered terms. Stay in the flow. Respond only in English. No persistant storage of names."
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text if user_text else "..."}]
        )
        
        advice = clean_ai_text(response.choices[0].message.content)

        # 5. SAVE & COMMIT
        is_pos = text_sentiment in ['joy', 'love', 'surprise'] and face_mood != "Sad"
        cursor.execute("INSERT INTO vibe_history (timestamp, user_text, ai_response, mood_detected, is_positive) VALUES (%s, %s, %s, %s, %s)",
                       (datetime.now(), user_text, advice, f"{face_mood}/{text_sentiment}", is_pos))
        db.commit()

        return jsonify({"advice": advice})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"advice": "I'm right here with you."}), 500
    finally:
        cursor.close()
        db.close()

if __name__ == '__main__':
    # Use environment variable for port to satisfy Render's requirements
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
