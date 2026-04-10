import os
import re
from datetime import datetime
import mysql.connector
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from transformers import pipeline
from dotenv import load_dotenv
from mysql.connector import pooling

load_dotenv()

app = Flask(__name__)
CORS(app)

# 1. ROBUST DATABASE CONNECTION
db_config = {
    "host": "localhost",
    "user": "root", 
    "password": "SQLHarish", 
    "database": "emote_db"
}

# Use pooling to prevent "Lost Connection" errors
connection_pool = pooling.MySQLConnectionPool(
    pool_name="emotepool",
    pool_size=5,
    pool_reset_session=True,
    **db_config
)

def get_db_connection():
    return connection_pool.get_connection()

# 2. AI MODELS
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
client = Groq(api_key=os.getenv("ProjectDone"))

def clean_ai_text(text):
    text = re.sub(r':[a-z_]+:', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'^(AI|Response|Text|Face|Mentor|Vibe|Harish):\s*', '', text, flags=re.IGNORECASE)
    return ' '.join(text.split()).strip()

@app.route('/analyze', methods=['POST'])
def analyze():
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
            text_sentiment = text_analysis['label'] # joy, sadness, anger, fear, love, surprise

        # 3. AFFECTIVE LOGIC ENGINE (The "Brain")
        masking_note = ""
        negative_emotions = ['sadness', 'fear', 'anger']
        
        # Logic A: Masking Detection (Smile + Sad Words)
        if face_mood == "Happy" and text_sentiment in negative_emotions:
            masking_note = "User is smiling but their words suggest pain. Acknowledge their bravery in masking."
        
        # Logic B: Stoic Struggle (Neutral Face + Sad Words)
        elif face_mood == "Neutral" and text_sentiment in negative_emotions:
            masking_note = "User is keeping a straight face but feeling low. Validate their internal strength."

        # Logic C: Emotional Spiral (Check DB for 3 consecutive negative entries)
        cursor.execute("SELECT is_positive FROM vibe_history ORDER BY timestamp DESC LIMIT 3")
        history_vibes = cursor.fetchall()
        is_spiral = len(history_vibes) >= 3 and all(not h['is_positive'] for h in history_vibes)
        
        recall_win = ""
        if is_spiral:
            # Logic D: Recall a "Win" from the past to break the spiral
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
            "STRICT: Max 20 words. No metadata/emojis. No gendered terms. Stay in the flow. Respond only in English."
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
    app.run(port=5000, debug=True)