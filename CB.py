import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
st.sidebar.image(r'WhatsApp Image 2024-07-17 at 22.45.42.jpeg')

s = st.sidebar.selectbox("Do You Know About CyberBulling HERE IT IS",("Cyberbulling",'CyberBulling_Forms'))
if s=='Cyberbulling':
        st.sidebar.write('''Cyberbullying refers to the use of digital technologies to harass, threaten, embarrass, or target another person.
                    It can take place through various online platforms and communication tools such as social media, text messages, emails, and
                    websites. Unlike traditional forms of bullying, cyberbullying can occur 24/7, reach a wide audience quickly, and
                    often leaves a digital footprint.''')
elif s=="CyberBulling_Forms":
        st.sidebar.write('''Forms of Cyberbullying:
                            Harassment: Sending mean or threatening messages to someone repeatedly.
                            Impersonation: Pretending to be someone else online to damage their reputation.
                            Exclusion: Deliberately excluding someone from an online group or activity.
                        Outing: Sharing someone's private information or secrets online without their consent.
                        Cyberstalking: Continuously monitoring and harassing someone online.
                        Trolling: Posting inflammatory or off-topic messages to provoke and upset others.''')

st.image(r"PB_Banner1.jpeg")
# Load the dataset
df = pd.read_csv(r'Cyberbulling.csv')

# Data Preprocessing
X = df[['comment']]
y = df['label']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Extraction and Model Training Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=5000), 'comment')
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Model Training
pipeline.fit(X_train, y_train)

# Model Prediction
y_pred = pipeline.predict(X_test)

# Accuracy and Classification Report
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Save the model and pipeline
with open('cyberbullying_model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

# Streamlit App Interface
st.title("Cyberbullying Detection")

# Display the dataset
#st.write("Dataset")
#st.write(df.head())

# Text input
input_text = st.text_area("Enter Your Received Message or comment from Unknown Persons:")



# Button to trigger prediction
if st.button("DETECT"):
    if input_text:
        # Load the model
        with open('cyberbullying_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        # Create input data
        input_data = pd.DataFrame({'comment': [input_text]})

        # Predict the label
        predicted_label = model.predict(input_data)[0]
        st.write(f"Predicted Label: {predicted_label}")

        # Display image based on the predicted label with 200x200 size
        if predicted_label == 'normal':
            st.image(r"normal.png", caption='Normal', width=200)
        elif predicted_label == 'offensive':
            st.image(r"offensive.png", caption='Offensive', width=200)
        elif predicted_label == 'hatespeech':
            st.image(r"hatespeech.png", caption='Hate Speech', width=200)

# Display accuracy and classification report
#st.write("Model Accuracy")
#st.write(accuracy)
#st.write("Classification Report")
#st.text(classification_report_str)
