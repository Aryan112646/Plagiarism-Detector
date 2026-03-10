import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('plagiarism.db')
    cursor = conn.cursor()
    # Create a table to store comparison history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file1_name TEXT,
            file2_name TEXT,
            similarity_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db() # Run this once when the app starts

@app.route('/')
def home():
    return "The Plagiarism Engine is Running with Database!"

@app.route('/compare', methods=['POST'])
def compare_files():
    # 1. Get files and names
    f1 = request.files['file1']
    f2 = request.files['file2']
    
    file1_content = f1.read().decode('utf-8')
    file2_content = f2.read().decode('utf-8')

    # 2. Logic: TF-IDF & Cosine Similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([file1_content, file2_content])
    score = cosine_similarity(tfidf_matrix)[0][1]
    percentage = round(score * 100, 2)

    # 3. SAVE TO DATABASE
    conn = sqlite3.connect('plagiarism.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (file1_name, file2_name, similarity_score) VALUES (?, ?, ?)",
        (f1.filename, f2.filename, percentage)
    )
    conn.commit()
    conn.close()
    
    return jsonify({"similarity": percentage})

@app.route('/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect('plagiarism.db')
    cursor = conn.cursor()
    # Get the last 10 comparisons
    cursor.execute("SELECT file1_name, file2_name, similarity_score, timestamp FROM history ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    
    history_data = []
    for row in rows:
        history_data.append({
            "f1": row[0],
            "f2": row[1],
            "score": row[2],
            "time": row[3]
        })
    return jsonify(history_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

