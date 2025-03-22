import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
SUPABASE_URL = 'https://klfiosrpujlpgsnxoqsg.supabase.co'
API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtsZmlvc3JwdWpscGdzbnhvcXNnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIyMDg5MzgsImV4cCI6MjA1Nzc4NDkzOH0.Y7JuLiyGq0nRQ_1wNNOcIwDNrfmAU5Xu4crHqMqW2bk'
ENDPOINT = f"{SUPABASE_URL}/rest/v1/Scrapped%20data"
headers = {
    "Authorization": API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json"
}
response = requests.get(ENDPOINT, headers=headers)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
    print(response.text)
    exit()
df.fillna(0, inplace=True)
le = LabelEncoder()
df['industries'] = le.fit_transform(df['industries'].astype(str))
X = df.drop(columns=['score'])  
y = df['score']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
joblib.dump(model, 'lead_scoring_model.pkl')
