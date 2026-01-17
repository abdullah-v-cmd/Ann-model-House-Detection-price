import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.models import Sequential, save_model, load_model

filename='//kaggle/input/house-price-prediction-dataset/csvdata.csv'
df=pd.read_csv(filename)
# print(df.head())
# print(df.isnull().sum())

X = df.drop('Price', axis=1)
y = df['Price']
# Separate numeric and categorical features
numeric_features = ['Area', 'No. of Bedrooms']
categorical_features = ['City', 'Location']
numeric_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scalar',StandardScaler())
])
categorical_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])
 
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get input shape for ANN
input_shape = X_train_processed.shape[1]

model = Sequential([
    Dense(128, activation='relu', input_dim=input_shape),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')  # linear output for regression
])

# Compile the ANN
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ==========================================
# 5️⃣ Train Model
# ==========================================
history = model.fit(
    X_train_processed, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=1
)

# ==========================================
# 6️⃣ Evaluate Performance
# ==========================================
y_pred = model.predict(X_test_processed).flatten()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Mean Squared Error: {mse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# ==========================================
# 7️⃣ Plot Training & Validation Loss
# ==========================================
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# ==========================================
# 8️⃣ Predict a New Sample
# ==========================================
sample = pd.DataFrame([{
    'City': 'Bangalore',
    'Area': 3776,
    'Location': 'Horamavu',
    'No. of Bedrooms': 4
}])

sample_processed = preprocessor.transform(sample)
pred_price = model.predict(sample_processed)
print(f"🏠 Predicted Price: {pred_price[0][0]:.2f}")