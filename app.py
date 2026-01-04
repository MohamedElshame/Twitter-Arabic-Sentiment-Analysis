from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_1_PATH = os.path.join(BASE_DIR, 'model1')
MODEL_2_PATH = os.path.join(BASE_DIR, 'model2')

# Labels for each model - FIXED ORDER to match training!
# Model 1: POS=0, NEG=1, OBJ=2, NEUTRAL=3
LABELS_MODEL_1 = {0: 'Positive', 1: 'Negative', 2: 'Objective', 3: 'Neutral'}
# Model 2: POS=0, NEG=1, OBJ=2  
LABELS_MODEL_2 = {0: 'Positive', 1: 'Negative', 2: 'Objective'}

# Load models
print('Loading models...')

tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_1_PATH)
model_1 = AutoModelForSequenceClassification.from_pretrained(MODEL_1_PATH)
model_1.eval()

tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_2_PATH)
model_2 = AutoModelForSequenceClassification.from_pretrained(MODEL_2_PATH)
model_2.eval()

print('Models loaded successfully!')


def predict_sentiment(text, model_choice):
    if model_choice == 'model1':
        tokenizer = tokenizer_1
        model = model_1
        labels = LABELS_MODEL_1
    else:
        tokenizer = tokenizer_2
        model = model_2
        labels = LABELS_MODEL_2

    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Get probabilities using softmax (from the model)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_class].item() * 100

    return {
        'sentiment': labels[predicted_class],
        'confidence': round(confidence, 1),
        'all_probs': {labels[i]: round(probabilities[0][i].item() * 100, 1) for i in labels}
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '').strip()
    model_choice = data.get('model', 'model1')

    if not text:
        return jsonify({'error': 'Please enter text to analyze'}), 400

    if len(text) > 1000:
        return jsonify({'error': 'Text too long, max 1000 characters'}), 400

    result = predict_sentiment(text, model_choice)
    
    # Log which model was used
    print(f"📊 Model: {model_choice.upper()} | Text: '{text[:50]}...' | Result: {result['sentiment']} ({result['confidence']}%)")
    
    return jsonify(result)


if __name__ == '__main__':
    print('\n' + '='*50)
    print('Server running at: http://127.0.0.1:5000')
    print('='*50 + '\n')
    app.run(debug=True, host='127.0.0.1', port=5000)
