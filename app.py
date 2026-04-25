from flask import Flask, request, jsonify, render_template
from sentiment_analysis_system import SentimentSystem
from collections import Counter
import re

app = Flask(__name__)

# Initialize and train model on startup
print("Initializing Sentiment Analysis System...")
system = SentimentSystem()
system.train_and_evaluate(verbose=False)
print("Model ready!")

def extract_keywords(text):
    # Very basic keyword extraction, skipping stopwords
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    from sentiment_analysis_system import STOPWORDS
    keywords = [w for w in words if w not in STOPWORDS]
    return keywords

def generate_improvements(negative_comments):
    """Rule-based generator for business improvements from negative text."""
    improvements = []
    
    # Simple keyword to suggestion mapping
    rules = {
        r'\b(price|cost|expensive|overpriced|money)\b': "Review pricing strategy or clearly communicate product value.",
        r'\b(service|support|staff|rude|unhelpful|customer)\b': "Enhance customer service training and ensure staff responsiveness.",
        r'\b(quality|cheap|broke|broken|garbage|rubbish|materials)\b': "Perform quality assurance checks on your products/materials.",
        r'\b(slow|wait|late|delayed|time)\b': "Optimize delivery or service processing times to reduce wait periods.",
        r'\b(app|crash|buggy|unusable)\b': "Investigate software bugs and improve app stability.",
        r'\b(food|cold|tasteless|flavor)\b': "Review food temperature controls and recipe consistency."
    }
    
    triggered_rules = set()
    for comment in negative_comments:
        text = comment.lower()
        for pattern, suggestion in rules.items():
            if re.search(pattern, text):
                if suggestion not in triggered_rules:
                    improvements.append(suggestion)
                    triggered_rules.add(suggestion)
    
    if not improvements and negative_comments:
        improvements.append("Investigate the general negative feedback to identify unspecified pain points.")
        
    return improvements

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = ""
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        text = file.read().decode('utf-8', errors='ignore')
    elif 'text' in request.form:
        text = request.form.get('text', '')
    elif request.is_json:
        text = request.json.get('text', '')
        
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    # Split by newlines or sentences for bulk analysis
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return jsonify({"error": "No valid text found"}), 400

    predictions = []
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    
    pos_keywords = []
    neg_keywords = []
    neu_keywords = []
    
    negative_comments = []

    for line in lines:
        res = system.predict(line)
        label = res['label']
        sentiment_counts[label] += 1
        
        predictions.append({
            "text": line,
            "label": label,
            "confidence": res['confidence']
        })
        
        kws = extract_keywords(line)
        if label == "positive":
            pos_keywords.extend(kws)
        elif label == "negative":
            neg_keywords.extend(kws)
            negative_comments.append(line)
        else:
            neu_keywords.extend(kws)

    # Percentages
    total = len(lines)
    percentages = {
        "positive": round((sentiment_counts["positive"] / total) * 100, 1),
        "negative": round((sentiment_counts["negative"] / total) * 100, 1),
        "neutral": round((sentiment_counts["neutral"] / total) * 100, 1)
    }

    # Top keywords globally and by sentiment
    all_keywords = pos_keywords + neg_keywords + neu_keywords
    top_global = [k for k, v in Counter(all_keywords).most_common(10)]
    
    stacked_keywords = {}
    for kw in top_global:
        stacked_keywords[kw] = {
            "positive": pos_keywords.count(kw),
            "negative": neg_keywords.count(kw),
            "neutral": neu_keywords.count(kw)
        }

    improvements = generate_improvements(negative_comments)

    return jsonify({
        "total_analyzed": total,
        "sentiment_counts": sentiment_counts,
        "percentages": percentages,
        "stacked_keywords": stacked_keywords,
        "improvements": improvements,
        "predictions": predictions[:100]  # limit payload size
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
