from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


app = Flask(__name__)
app.secret_key = "super_secret_key"
from urllib.parse import quote_plus

password = quote_plus("hemu@123")

# ---------------- MYSQL CONFIG ----------------
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://root:{password}@localhost/sentiment_analysis'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


stress_words = [
'stress','stressed','overwhelmed','pressure','burnout','tired','exhausted','fatigue',
'deadline','workload','burden','strained','restless','tense','frustrated','irritated',
'panic','rush','hurry','chaos','hectic','uneasy','disturbed','troubled','drained',
'overworked','nagging','tight','mental load','fatigued','strain','agitated','distress',
'confused','overthinking','worried','pressureful','overloaded','strained mind','tension',
'imbalance','discomfort','trapped','uneasy feeling','mental stress','work stress',
'life stress','heavy','too much', "can't handle",'pressure build','burned out'
]

# 🔹 Anxiety Keywords
anxiety_words = [
'anxious','anxiety','worried','fear','scared','nervous','panic','uneasy','restless',
'overthinking','doubt','insecure','uncertain','tense','afraid','phobia','shaking',
'sweating','heart racing','paranoid','fearful','panic attack','nervousness',
'apprehensive','concerned','uneasiness','disturbed','trembling','fearful thoughts',
'worrying','obsessing','anticipation','dread','mental fear','panic mode','hyper',
'restlessness','shaky','fear of failure','fear of future','fear of people',
'claustrophobic','social anxiety','general anxiety','irrational fear',
'constant worry','uneasy mind','fear inside','over alert','panic feeling'
]

# 🔹 Depression Keywords
depression_words = [
'sad','sadness','depressed','depression','hopeless','empty','lonely','worthless',
'helpless','tired','no energy','loss of interest','crying','low','down','miserable',
'heartbroken','guilty','regret','pain','hurt','dark','numb','isolated','withdrawn',
'unmotivated','hopelessness','despair','failure','worthlessness','low mood','blue',
'broken','lost','grief','no hope','nothing matters','emotional pain','inner pain',
'self doubt','negative thoughts','low confidence','mental pain','cry','alone',
'feeling low','deep sadness','unhappy','empty inside','dead inside'
]

# 🔹 Suicidal Keywords
suicidal_words = [
'suicide','kill myself','end my life','want to die','die','no reason to live',
'give up','ending it','better off dead','self harm','cut myself','hurt myself',
'overdose','jump off','hang myself','life is pointless','no hope left',
'i want to disappear','i hate my life','leave this world','not worth living',
'escape life','death thoughts','suicidal thoughts','thinking of dying',
'can’t go on','cant go on','want it to end','life is over','i am done',
'goodbye forever','nothing matters anymore','final decision','end everything',
'tired of life','no future','stop living','give up life','pain is too much',
'i quit life','last message','end suffering','permanent sleep','leave forever',
'no purpose','ready to die','end this pain'
]

# ---------------- MODEL ----------------
class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), default='user')

    def __repr__(self):
        return f"<User {self.email}>"


# ---------------- CREATE TABLE ----------------
with app.app_context():
    db.create_all()


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# notebook_path = r'C:\MH_Sentiment_Project\notebooks'
notebook_path = r'.\code'
# h5_file = r'C:\MH_Sentiment_Project\MH_Sentiment_Project\notebooks'
model_path = os.path.join(notebook_path, 'mental_health_model.h5')
tokenizer_path = os.path.join(notebook_path, 'tokenizer.pkl')
label_encoder_path = os.path.join(notebook_path, 'label_encoder.pkl')

try:
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
        print('tokenizer ', tokenizer)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    model_loaded = True
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None
    tokenizer = None
    label_encoder = None
    model_loaded = False

# ---------------- REGISTER ----------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        role = request.form.get('role', 'user')  # default user

        # Check existing email
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered!")
            return redirect(url_for('register'))

        new_user = User(
            name=name,
            email=email,
            # password=password,
            role=role
        )

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['id'] = user.id
            session['name'] = user.name
            session['email'] = user.email
            session['role'] = user.role

            # 🔥 REDIRECT TO INDEX AFTER LOGIN
            return redirect(url_for('index'))

        else:
            flash("Invalid email or password!")

    return render_template('login.html')
# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template('home.html')
    # if 'email' not in session:
    #     return redirect(url_for('login'))

    # members = []
    # if session['role'] == 'admin':
    #     members = User.query.all()

    # return render_template(
    #     'home.html',
    #     name=session['name'],
    #     role=session['role'],
    #     members=members
    # )

@app.route('/admin')
def admin():

    # Check login
    if 'email' not in session:
        return redirect(url_for('login'))

    # Allow only admin
    if session.get('role') != 'admin':
        flash("Access denied. Admin only.")
        return redirect(url_for('index'))

    # Fetch all users
    users = User.query.all()

    return render_template('admin.html', users=users)



@app.route('/index')
def index():
    if 'email' not in session:
        return redirect(url_for('login'))

    return render_template('index.html')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ---------------- PREDICT API ----------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in session:
        return redirect(url_for('login'))

    text = request.form['text_input']
    print("text", text)
    processed = preprocess_text(text)
    print(processed)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    pred = model.predict(padded)
    pred_class = np.argmax(pred, axis=1)
    sentiment = label_encoder.inverse_transform(pred_class)[0]
    print
    if sentiment == 'Normal':
        if any(word in text for word in suicidal_words):
            sentiment =  "Suicidal"
        elif any(word in text for word in depression_words):
            sentiment =  "Depression"
        elif any(word in text for word in anxiety_words):
            sentiment =  "Anxiety"
        elif any(word in text for word in stress_words):
            sentiment =  "Stress"

    return render_template(
        'index.html',
        prediction=sentiment,
        user_text=text
    )



if __name__ == "__main__":
    app.run()
