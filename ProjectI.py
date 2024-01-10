# Import necessary libraries
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the Matplotlib backend to Agg for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK data for stopwords
nltk.download('stopwords')

# Load the dataset
emails_filepath = 'messages.csv'
emails_df = pd.read_csv(emails_filepath)

# Clean the dataset by removing rows with missing values
cleaned_emails_df = emails_df.dropna().copy()

# Text preprocessing: remove punctuation, convert to lowercase, and remove stopwords
stop_words = set(stopwords.words('english'))
cleaned_emails_df['clean_subject'] = cleaned_emails_df['subject'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())
cleaned_emails_df['clean_message'] = cleaned_emails_df['message'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())
cleaned_emails_df['clean_subject'] = cleaned_emails_df['clean_subject'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
cleaned_emails_df['clean_message'] = cleaned_emails_df['clean_message'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

# Save the cleaned data to a CSV file
cleaned_data_filepath = 'cleaned_emails.csv'
cleaned_emails_df.to_csv(cleaned_data_filepath, index=False)

# Convert the cleaned message text into numerical features using CountVectorizer
vectorizer = CountVectorizer()
features_matrix = vectorizer.fit_transform(cleaned_emails_df['clean_message'])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_matrix, cleaned_emails_df['label'], test_size=0.3, random_state=42)

# Train a Multinomial Naive Bayes classifier on the training data
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict labels for the test data
predictions = nb_classifier.predict(X_test)

# Generate a confusion matrix from the predictions
confusion_mtx = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap="Blues")
plt.title('Confusion Matrix for Email Classification')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
confusion_mtx_filepath = "email_confusion_mtx.png"
plt.savefig(confusion_mtx_filepath, dpi=300)
plt.close()

# Print the accuracy and save the classification report to a text file
classification_accuracy = metrics.accuracy_score(y_test, predictions)
print(f"Classification Accuracy: {classification_accuracy:.2%}")
classification_report_filepath = "email_classification_report.txt"
with open(classification_report_filepath, "w") as report_file:
    class_report = metrics.classification_report(y_test, predictions)
    report_file.write(class_report)
    report_file.write("\nConfusion Matrix:\n")
    report_file.write(str(confusion_mtx))

# Create and save a pie chart visualizing the proportion of spam to non-spam emails
email_labels = ['Non-Spam', 'Spam']
email_counts = cleaned_emails_df['label'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(email_counts, labels=email_labels, colors=['skyblue', 'orange'], autopct='%1.1f%%', startangle=140, explode=(0.1, 0))
plt.title('Distribution of Spam vs Non-Spam Emails')
email_distribution_filepath = "email_distribution_pie.png"
plt.savefig(email_distribution_filepath, dpi=300)
plt.close()

# Define a function to visualize the top terms in the emails
def visualize_top_terms(text_series, chart_title, num_terms=20, chart_filename="top_terms_visual.png"):
    # Generate term frequency matrix
    vectorizer = CountVectorizer(stop_words='english')
    term_matrix = vectorizer.fit_transform(text_series)
    term_sum = term_matrix.sum(axis=0)
    term_freqs = [(word, term_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    term_freqs.sort(key=lambda x: x[1], reverse=True)
    top_terms = pd.DataFrame(term_freqs[:num_terms], columns=['Term', 'Frequency'])

    # Plot a bar chart for the top terms
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Term', data=top_terms)
    plt.title(chart_title)
    plt.xlabel('Occurrence Count')
    plt.ylabel('Significant Terms')
    plt.savefig(chart_filename, dpi=300)
    plt.close()

# Visualize and save the top terms in spam and non-spam messages
visualize_top_terms(cleaned_emails_df[cleaned_emails_df['label'] == 1]['clean_message'], 'Frequent Terms in Spam Emails', chart_filename="spam_terms_visual.png")
visualize_top_terms(cleaned_emails_df[cleaned_emails_df['label'] == 0]['clean_message'], 'Frequent Terms in Non-Spam Emails', chart_filename="non_spam_terms_visual.png")
