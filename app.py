from flask import Flask, render_template, request
import joblib

# ──────────────────────────────────────────────────────────────
# INITIALIZE FLASK APP
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)

# ──────────────────────────────────────────────────────────────
# LOAD TRAINED MODEL
# ──────────────────────────────────────────────────────────────
model = joblib.load('model.pkl')

# ──────────────────────────────────────────────────────────────
# ROUTE 1: HOME PAGE
# ──────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# ──────────────────────────────────────────────────────────────
# ROUTE 2: PREDICT
# ──────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form.get('email_text', '').strip()

    if not email_text:
        return render_template('index.html',
                               prediction=None,
                               error="⚠️ Please enter an email to analyse.",
                               email_text=email_text)

    # Predict using the loaded model
    result = model.predict([email_text])[0]
    probability = model.predict_proba([email_text])[0]

    spam_confidence  = round(probability[1] * 100, 2)
    ham_confidence   = round(probability[0] * 100, 2)

    if result == 1:
        prediction  = "HARMFUL"
        verdict     = "🚨 This email is SPAM (Harmful)"
        description = "This email appears to be spam or a phishing attempt. Do NOT click any links or share personal information."
        badge_class = "harmful"
        confidence  = spam_confidence
    else:
        prediction  = "HARMLESS"
        verdict     = "✅ This email is SAFE (Harmless)"
        description = "This email appears to be legitimate and safe. It does not show signs of spam or phishing."
        badge_class = "harmless"
        confidence  = ham_confidence

    return render_template('index.html',
                           prediction=prediction,
                           verdict=verdict,
                           description=description,
                           badge_class=badge_class,
                           confidence=confidence,
                           spam_confidence=spam_confidence,
                           ham_confidence=ham_confidence,
                           email_text=email_text)

# ──────────────────────────────────────────────────────────────
# RUN APP
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)