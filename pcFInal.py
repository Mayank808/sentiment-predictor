# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# %%
large_train = "./424_F2024_Final_PC_large_train_v1.csv"
test_path = "./424_F2024_Final_PC_test_without_response_v1.csv"
model_path = "my_model_500epoch.h5"

df_train = pd.read_csv(large_train)
df_test = pd.read_csv(test_path)

# %%
df_train.sample(5)

# %%
df_test.sample(5)


# %%

nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):
        return 'Unknown'
    text = text.encode('ascii', 'ignore').decode()  
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower().strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Preprocess the training and test data
def preprocess_data(df_train, df_test):
    # Clean text columns
    for col in ['pros', 'cons', 'headline']:
        df_train[col] = df_train[col].apply(clean_text)
        df_test[col] = df_test[col].apply(clean_text)

    # Combine text columns for vectorization
    combined_text_train = df_train['pros'] + " " + df_train['cons'] + " " + df_train['headline']
    combined_text_test = df_test['pros'] + " " + df_test['cons'] + " " + df_test['headline']

    # Initialize the TF-IDF vectorizer with an increased max_features
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjusted to capture more information

    # Fit and transform the training data, transform the test data
    combined_tfidf_train = tfidf_vectorizer.fit_transform(combined_text_train).toarray()
    combined_tfidf_test = tfidf_vectorizer.transform(combined_text_test).toarray()

    # Handle missing or empty values in 'job_title' and 'firm'
    df_train['job_title'] = df_train['job_title'].fillna('Unknown')
    df_test['job_title'] = df_test['job_title'].fillna('Unknown')
    df_train['firm'] = df_train['firm'].fillna('Unknown')
    df_test['firm'] = df_test['firm'].fillna('Unknown')

    # Encode 'firm' using LabelEncoder
    label_encoder_firm = LabelEncoder()
    all_firm_values = pd.concat([df_train['firm'], df_test['firm']])
    label_encoder_firm.fit(all_firm_values)
    df_train['firm_encoded'] = label_encoder_firm.transform(df_train['firm'])
    df_test['firm_encoded'] = label_encoder_firm.transform(df_test['firm'])

    # Convert 'rating' to a numeric target for regression
    y_train = df_train['rating'].values
    y_test = df_test['rating'].values

    # Combine all the features for both training and test sets
    X_train = np.hstack([
        combined_tfidf_train,
        df_train[['firm_encoded']].values
    ]).astype('float32')

    X_test = np.hstack([
        combined_tfidf_test,
        df_test[['firm_encoded']].values
    ]).astype('float32')

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Return processed features and labels
    return X_train, X_test, y_train, y_test

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(df_train, df_test)

# %%
X_train.shape, X_test.shape

# %%
X_train[:5]

# %%
# Build the model with adjusted architecture
model = Sequential([
    Dense(256, input_dim=X_train.shape[1], kernel_regularizer=l2(0.001)),  # Added L2 regularization
    BatchNormalization(),
    LeakyReLU(alpha=0.1),  # LeakyReLU for better gradient flow
    Dropout(0.25),  # Fine-tuned dropout rate

    Dense(128, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),

    Dense(64, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.15),

    Dense(32, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),

    Dense(1, activation='linear')  # Linear activation for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mse'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_mse', patience=30, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_mse', factor=0.5, patience=15, min_lr=0.00001)

# Fit the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=500,  # Increased epochs for better learning
    batch_size=64,  # Option to try other sizes
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

# Save the trained model
model.save(model_path)  # Save entire model to HDF5 format

# %%
model = load_model(model_path) 

# %%
y_pred = model.predict(X_train)

print("Predictions on the test set:")
print(y_pred)

# %%
# Calculate Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train, y_pred)

# Calculate R-squared
r2_train = r2_score(y_train, y_pred)

# Print the results
print(f"Training MSE: {mse_train}")
print(f"Training R-squared: {r2_train}")

# %%
y_pred_test = model.predict(X_test)
print(y_pred_test)

# %%
# Step 1: Define your information
student_id = '20952031' 
anonymized_name = '808' 
prediction_accuracy = r2_train  
algorithm_name = 'Neural Network Model' 

data = [
    [student_id],
    [anonymized_name],
    [prediction_accuracy],
    [algorithm_name]
] + [pred for pred in y_pred_test]


df = pd.DataFrame(data)

df.to_csv('final_perdictions_learn1.csv', header=False, index=False)

print("CSV file created successfully.")

# %%
print(y_pred_test)

# %%
# Create the Kaggle-style DataFrame
y_data_kaggle = []
for pred in y_pred_test:
    for val in pred:
        y_data_kaggle.append(val)

data = {
    "ID_num": range(1, len(y_data_kaggle) + 1),
    "prediction": y_data_kaggle
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('kaggle_predictions1.csv', index=False)

print("CSV file created successfully in Kaggle format.")

# %%
model.summary()

# %%
from keras.utils import plot_model

# Visualize model architecture
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

# %%

# Display the model structure image
plt.figure(figsize=(10, 35))
img = plt.imread('model_structure.png')
plt.imshow(img)
plt.axis('off')
plt.show()

# %%
from wordcloud import WordCloud

for star_rating in range(1, 6):
    # Combine all 'pros' text for the current star rating
    review_text = ' '.join(df_train[df_train['rating'] == star_rating]['pros'].tolist())
    
    # Create a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(review_text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {star_rating}-Star Reviews')
    plt.axis('off')  # Hide axes for better visualization
    plt.show()

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Encode categorical features
label_encoder = LabelEncoder()
df_train['firm_encoded'] = label_encoder.fit_transform(df_train['firm'].fillna(''))
df_train['job_title_encoded'] = label_encoder.fit_transform(df_train['job_title'].fillna(''))

# Define features and target variable
feature_columns = ['year_review', 'firm_encoded', 'job_title_encoded']
target_column = 'rating'  # Predicting the 'rating'

# Prepare data for training and testing
X = df_train[feature_columns]
y = df_train[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model
random_forest = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
random_forest.fit(X_train, y_train)

# Extract feature importance scores
feature_importances = random_forest.feature_importances_

# Visualize feature importance
plt.figure(figsize=(8, 6))
plt.barh(feature_columns, feature_importances, color='skyblue')
plt.title("Feature Importance for Rating Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature Names")
plt.tight_layout()  # Adjust layout to fit title and labels
plt.show()

# %%
correlation_matrix = X.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))  # Adjust figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix for Features')
plt.show()

# %%
# Plot the distribution of each feature in both the training and test datasets
# Visualize the distribution of each feature in the training and testing datasets
for feature_name in feature_columns:  # Renamed 'features' to 'feature_columns' for consistency
    plt.figure(figsize=(8, 6))
    
    # Plot the distribution of the feature in the training data
    plt.hist(
        X_train[feature_name], 
        bins=30, 
        alpha=0.7, 
        color='blue', 
        label='Training Data', 
        density=True
    )
    
    # Plot the distribution of the feature in the test data
    plt.hist(
        X_test[feature_name], 
        bins=30, 
        alpha=0.7, 
        color='red', 
        label='Testing Data', 
        density=True
    )
    
    # Add title and labels
    plt.title(f"Distribution of '{feature_name}' in Training vs Testing Data")
    plt.xlabel(f"{feature_name}")
    plt.ylabel("Density")
    plt.legend()  # Show the legend to distinguish between Train and Test data
    plt.tight_layout()  # Ensure the layout fits nicely
    plt.show()


# %%
# Predict ratings for both training and test sets
train_predictions = random_forest.predict(X_train)
test_predictions = random_forest.predict(X_test)

# Plot the distribution of actual vs. predicted ratings for training and test datasets
plt.figure(figsize=(12, 6))

# Plot for the training data
plt.subplot(1, 2, 1)
plt.hist(y_train, bins=30, alpha=0.7, color='blue', label='Actual (Training)', density=True)
plt.hist(train_predictions, bins=30, alpha=0.7, color='red', label='Predicted (Training)', density=True)
plt.title("Actual vs. Predicted Ratings: Training Data")
plt.xlabel("Rating")
plt.ylabel("Density")
plt.legend()

# Plot for the test data
plt.subplot(1, 2, 2)
plt.hist(y_test, bins=30, alpha=0.7, color='blue', label='Actual (Testing)', density=True)
plt.hist(test_predictions, bins=30, alpha=0.7, color='red', label='Predicted (Testing)', density=True)
plt.title("Actual vs. Predicted Ratings: Testing Data")
plt.xlabel("Rating")
plt.ylabel("Density")
plt.legend()

# Adjust layout for better visualization
plt.tight_layout()
plt.show()


# %%
# Calculate residuals (errors) for training and testing datasets
train_residuals = y_train - train_predictions
test_residuals = y_test - test_predictions

# Visualize the distribution of residuals
plt.figure(figsize=(12, 6))

# Residual plot for training data
plt.subplot(1, 2, 1)
plt.scatter(y_train, train_residuals, alpha=0.5, color='blue')
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Reference line at 0
plt.title("Actual Ratings vs Residuals: Training Data")
plt.xlabel("Actual Rating")
plt.ylabel("Residual (Error)")

# Residual plot for testing data
plt.subplot(1, 2, 2)
plt.scatter(y_test, test_residuals, alpha=0.5, color='red')
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Reference line at 0
plt.title("Actual Ratings vs Residuals: Testing Data")
plt.xlabel("Actual Rating")
plt.ylabel("Residual (Error)")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()


# %%
def get_top_phrases_by_rating(df, rating_threshold):
    subset = df[df['rating'] >= rating_threshold]
    combined_text = ' '.join(subset['pros'] + ' ' + subset['cons'] + ' ' + subset['headline'])
    word_freq = pd.Series(combined_text.split()).value_counts().head(10)
    return word_freq

high_rating_phrases = get_top_phrases_by_rating(df_train, 4)
low_rating_phrases = get_top_phrases_by_rating(df_train, 2)

print("Top phrases in high-rated firms:")
print(high_rating_phrases)
print("\nTop phrases in low-rated firms:")
print(low_rating_phrases)

# %%
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# Load data
df = pd.read_csv(large_train)

# Categorize firms by average rating (low: 1-2, medium: 3-4, high: 5)
df['rating_group'] = pd.cut(df['rating'], bins=[0, 2, 4, 5], labels=['Low', 'Medium', 'High'])

# Function to extract top words/themes
def extract_themes(text_column, top_n=10):
    vectorizer = CountVectorizer(stop_words='english')
    word_counts = vectorizer.fit_transform(text_column.dropna())
    word_sum = word_counts.sum(axis=0)
    words_freq = [(word, word_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
    return words_freq

# Analyze themes for pros and cons by rating group
themes = {}
for group in df['rating_group'].unique():
    group_data = df[df['rating_group'] == group]
    pros_themes = extract_themes(group_data['pros'], top_n=10)
    cons_themes = extract_themes(group_data['cons'], top_n=10)
    themes[group] = {'pros': pros_themes, 'cons': cons_themes}

# Display results in a table
for group, data in themes.items():
    print(f"\n=== {group} Rated Firms ===")
    print("Top Pros:", data['pros'])
    print("Top Cons:", data['cons'])

# Optional: Create word clouds for visualization
def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()

for group in df['rating_group'].unique():
    group_data = df[df['rating_group'] == group]
    plot_wordcloud(group_data['pros'].dropna(), f"{group} Rated Firms - Pros")
    plot_wordcloud(group_data['cons'].dropna(), f"{group} Rated Firms - Cons")



