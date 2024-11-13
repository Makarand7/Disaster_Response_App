import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

def load_data(database_filepath):
    """
    Load data from the SQLite database.
    
    Args:
    database_filepath (str): Filepath for the SQLite database containing the cleaned data.
    
    Returns:
    X (pd.Series): Messages for training.
    Y (pd.DataFrame): Labels for the categories.
    category_names (list): List of category names for classification.
    """
    # Load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('YourTableName', engine)
    
    # Define feature and target variables
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and process text data.
    
    Args:
    text (str): Text message to be tokenized.
    
    Returns:
    clean_tokens (list): List of cleaned tokens.
    """
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    
    # Lemmatize and clean tokens
    lemmatizer = nltk.WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok not in stop_words]
    
    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline and perform grid search.
    
    Returns:
    model (GridSearchCV): Grid search model object.
    """
    # Build pipeline with text processing and classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Set grid search parameters
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }
    
    # Initialize grid search with pipeline and parameters
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=1, cv=3)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model by printing classification report for each category.
    
    Args:
    model: Trained machine learning model.
    X_test (pd.Series): Test features.
    Y_test (pd.DataFrame): True labels for test data.
    category_names (list): List of category names.
    """
    # Predict on test data
    Y_pred = model.predict(X_test)
    
    # Print classification report for each category
    for i, col in enumerate(category_names):
        print(f'Category: {col}\n', classification_report(Y_test[col], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    
    Args:
    model: Trained machine learning model.
    model_filepath (str): Filepath for saving the pickle file.
    """
    # Ensure the directory exists
    model_dir = os.path.dirname(model_filepath)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save model to pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function to execute data loading, model building, training,
    evaluation, and saving.
    """
    # Check if correct number of arguments is provided
    if len(sys.argv) != 3:
        print('Error: Please provide the filepath of the disaster messages database and model pickle file.')
        sys.exit(1)

    database_filepath, model_filepath = sys.argv[1:]
    
    # Check if the database file exists
    if not os.path.exists(database_filepath):
        print(f"Error: The file '{database_filepath}' does not exist.")
        sys.exit(1)

    print('Loading data...\n    DATABASE:', database_filepath)
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    print('Building model...')
    model = build_model()
    
    print('Training model...')
    try:
        model.fit(X_train, Y_train)
    except Exception as e:
        print(f"Error in training the model: {e}")
        sys.exit(1)
    
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL:', model_filepath)
    save_model(model.best_estimator_, model_filepath)  # Save the best model after grid search

    print('Trained model saved!')

if __name__ == '__main__':
    main()
