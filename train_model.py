import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ──────────────────────────────────────────────────────────────
# DATASET  (Spam = 1 → Harmful | Ham = 0 → Harmless)
# ──────────────────────────────────────────────────────────────
data = {
    'email': [
        # ── SPAM (Harmful) ──────────────────────────────────
        "Congratulations! You have won a $1,000 Walmart gift card. Click here now!",
        "URGENT: Your account has been compromised. Verify now to avoid suspension.",
        "Free money! Claim your prize. Limited time offer. Click the link below.",
        "You are selected for a cash reward. Send your bank details immediately.",
        "Buy cheap medicines online! No prescription needed. 90% discount today!",
        "Make $5000 from home daily! No experience needed. Join now for FREE!",
        "Your PayPal account is limited. Confirm your identity here immediately.",
        "Hot singles near you! Click here to meet them. 100% FREE registration.",
        "WINNER: You have been chosen as a lucky winner. Claim $10,000 prize now!",
        "Lowest mortgage rates! Apply now and save thousands. No credit check!",
        "Nigerian prince needs your help to transfer $10 million. Share profits!",
        "Earn unlimited income working from home. No skills required. Join today!",
        "Your email has won the UK lottery. Send details to claim your winnings.",
        "Lose 30 pounds in 30 days with this miracle pill. Buy 2 get 3 free!",
        "Get rich quick scheme - make $1000 per day. 100% guaranteed success!",
        "Click here to unsubscribe and win a free iPhone 15 Pro. Limited stock!",
        "Your bank account has suspicious activity. Login here to verify: bit.ly",
        "Exclusive deal ONLY for you! 95% OFF luxury watches. Order in next 5 min!",
        "Cheap Viagra online! No prescription needed. Discreet shipping worldwide.",
        "You owe taxes! Immediate payment required or face arrest. Call now!!!",
        "FREE credit score check! Enter your SSN and bank info to get started.",
        "Claim your inheritance of $4.5 million. Reply with your bank account now.",
        "Weight loss miracle! Doctors hate this trick. Lose 20 lbs in 1 week!",
        "FINAL NOTICE: Your subscription expired. Update billing info immediately.",
        "Crypto investment opportunity! 500% returns guaranteed in 24 hours. Join!",

        # ── HAM (Harmless) ───────────────────────────────────
        "Hey, can we schedule a meeting for next Tuesday at 3 PM?",
        "Please find the attached report for your review. Let me know your thoughts.",
        "Happy Birthday! Hope you have a wonderful day filled with joy.",
        "Reminder: Your dentist appointment is tomorrow at 10 AM. Please confirm.",
        "Thanks for the help yesterday. The project is going smoothly now.",
        "Can you please send me the updated version of the presentation?",
        "Your order has been shipped! Expected delivery: 3-5 business days.",
        "Team lunch is confirmed for Friday at 1 PM. See you all there!",
        "Please review the attached document and share your feedback by Thursday.",
        "Hi, I will be out of office from Monday to Wednesday. Contact John instead.",
        "Your subscription renewal is due next month. No action needed now.",
        "The quarterly results are attached. Great job everyone on the team!",
        "Can we discuss the project timeline during tomorrow's standup meeting?",
        "Your flight booking is confirmed. Check-in opens 24 hours before departure.",
        "Thank you for attending our webinar. Here is the recording link for you.",
        "I have reviewed your code. There are a few minor changes needed. See comments.",
        "Library books are due for return by end of this week. Please return them.",
        "The meeting agenda has been updated. Please check the shared Google Doc.",
        "Welcome to the team! Your onboarding schedule is attached to this email.",
        "Please find the invoice for last month's services. Payment due in 30 days.",
        "Your password was changed successfully. If this was not you, contact support.",
        "Mom's birthday party is on Saturday at 6 PM. Don't forget to bring the cake!",
        "The conference call has been rescheduled to 4 PM. Updated invite is attached.",
        "Great work on the presentation yesterday! The client was very impressed.",
        "Your annual performance review is scheduled for next Monday at 2 PM.",
    ],
    'label': [
        # Spam labels (1)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # Ham labels (0)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
}

df = pd.DataFrame(data)

print(f"Dataset Size  : {len(df)} emails")
print(f"Spam (Harmful): {df['label'].sum()} emails")
print(f"Ham (Harmless): {(df['label'] == 0).sum()} emails")
print()

# ──────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['email'], df['label'], test_size=0.2, random_state=42
)

# ──────────────────────────────────────────────────────────────
# PIPELINE  (TF-IDF + Naive Bayes)
# ──────────────────────────────────────────────────────────────
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))),
    ('classifier', MultinomialNB(alpha=0.1))
])

# ──────────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────────
model.fit(X_train, y_train)

# ──────────────────────────────────────────────────────────────
# EVALUATE
# ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Harmless (Ham)', 'Harmful (Spam)']))

# ──────────────────────────────────────────────────────────────
# SAVE MODEL
# ──────────────────────────────────────────────────────────────
joblib.dump(model, 'model.pkl')
print("✅ Model Trained and Saved Successfully as model.pkl")