"""
Netflix Global Content Analysis
===============================
This module performs comprehensive data science analysis on Netflix content data,
including country-wise genre dominance, musical content scoring, and time series analysis.

Required Libraries: numpy, pandas, matplotlib, seaborn, sklearn, scipy.stats
"""
"""
Movie Rating Prediction Module
==============================
This module implements machine learning models (Decision Tree and Random Forest)
to predict movie ratings (G, PG, PG-13, R) based on features like:
- Duration
- Genre
- Country
- Release Year
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings

warnings.filterwarnings('ignore')


def extract_duration_minutes(duration_str):
    """
    Extract numeric duration in minutes from duration string.
    Handles formats like '90 min' for movies and '2 Seasons' for TV shows.
    """
    if pd.isna(duration_str) or duration_str == '':
        return np.nan
    
    duration_str = str(duration_str).lower().strip()
    
    if 'min' in duration_str:
        try:
            return int(duration_str.replace('min', '').strip())
        except ValueError:
            return np.nan
    elif 'season' in duration_str:
        try:
            seasons = int(duration_str.replace('seasons', '').replace('season', '').strip())
            return seasons * 10 * 30
        except ValueError:
            return np.nan
    return np.nan


def extract_release_year(df):
    """
    Extract release year from the dataframe.
    Uses 'release_year' column if available.
    """
    if 'release_year' in df.columns:
        return pd.to_numeric(df['release_year'], errors='coerce')
    return pd.Series([np.nan] * len(df))


def get_primary_genre(listed_in):
    """
    Extract the primary (first) genre from listed_in column.
    """
    if pd.isna(listed_in) or listed_in == '':
        return 'Unknown'
    return str(listed_in).split(',')[0].strip()


VALID_RATINGS = ['G', 'PG', 'PG-13', 'R']


def prepare_features_for_ml(df):
    """
    Prepare features for machine learning model.
    Uses actual rating column from the Netflix dataset.
    Filters to only include movies with MPAA ratings (G, PG, PG-13, R).
    """
    df_ml = df.copy()
    
    df_ml['duration_minutes'] = df_ml['duration'].apply(extract_duration_minutes)
    
    df_ml['release_year'] = extract_release_year(df_ml)
    
    df_ml['primary_genre'] = df_ml['listed_in'].apply(get_primary_genre)
    
    if 'primary_country' not in df_ml.columns:
        df_ml['primary_country'] = df_ml['country'].apply(
            lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown'
        )
    
    df_ml = df_ml[df_ml['rating'].isin(VALID_RATINGS)]
    df_ml['rating_label'] = df_ml['rating']
    
    df_ml = df_ml.dropna(subset=['duration_minutes', 'release_year'])
    
    return df_ml


def encode_features(df, genre_encoder=None, country_encoder=None, fit=True):
    """
    Encode categorical features using LabelEncoder.
    Returns encoded dataframe and fitted encoders.
    """
    df_encoded = df.copy()
    
    if fit:
        genre_encoder = LabelEncoder()
        country_encoder = LabelEncoder()
        df_encoded['genre_encoded'] = genre_encoder.fit_transform(df_encoded['primary_genre'].astype(str))
        df_encoded['country_encoded'] = country_encoder.fit_transform(df_encoded['primary_country'].astype(str))
    else:
        df_encoded['genre_encoded'] = df_encoded['primary_genre'].apply(
            lambda x: genre_encoder.transform([str(x)])[0] if str(x) in genre_encoder.classes_ else -1
        )
        df_encoded['country_encoded'] = df_encoded['primary_country'].apply(
            lambda x: country_encoder.transform([str(x)])[0] if str(x) in country_encoder.classes_ else -1
        )
    
    return df_encoded, genre_encoder, country_encoder


def train_models(df_encoded, test_size=0.2, random_state=42):
    """
    Train Decision Tree and Random Forest models.
    Returns trained models, encoders, and evaluation metrics.
    """
    feature_cols = ['duration_minutes', 'genre_encoded', 'country_encoded', 'release_year']
    X = df_encoded[feature_cols]
    y = df_encoded['rating_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    rating_encoder = LabelEncoder()
    y_train_encoded = rating_encoder.fit_transform(y_train)
    y_test_encoded = rating_encoder.transform(y_test)
    
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state
    )
    dt_model.fit(X_train, y_train_encoded)
    dt_predictions = dt_model.predict(X_test)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train_encoded)
    rf_predictions = rf_model.predict(X_test)
    
    dt_metrics = calculate_metrics(y_test_encoded, dt_predictions, rating_encoder)
    rf_metrics = calculate_metrics(y_test_encoded, rf_predictions, rating_encoder)
    
    return {
        'decision_tree': {
            'model': dt_model,
            'metrics': dt_metrics,
            'predictions': dt_predictions,
            'y_test': y_test_encoded
        },
        'random_forest': {
            'model': rf_model,
            'metrics': rf_metrics,
            'predictions': rf_predictions,
            'y_test': y_test_encoded
        },
        'rating_encoder': rating_encoder,
        'feature_cols': feature_cols,
        'X_test': X_test,
        'y_test': y_test
    }


def calculate_metrics(y_true, y_pred, encoder):
    """
    Calculate classification metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classes': encoder.classes_
    }


def create_confusion_matrix_plot(conf_matrix, classes, title="Confusion Matrix"):
    """
    Create a confusion matrix heatmap visualization.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )
    
    ax.set_xlabel('Predicted Rating', fontsize=12)
    ax.set_ylabel('Actual Rating', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_feature_importance_plot(model, feature_names, title="Feature Importance"):
    """
    Create a bar plot showing feature importance from the model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(feature_names)))[::-1]
    
    bars = ax.barh(
        [feature_names[i] for i in indices[::-1]],
        [importances[i] for i in indices[::-1]],
        color=colors
    )
    
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    for bar, imp in zip(bars, [importances[i] for i in indices[::-1]]):
        ax.text(imp + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def create_model_comparison_plot(dt_metrics, rf_metrics):
    """
    Create a bar plot comparing Decision Tree and Random Forest metrics.
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    dt_values = [dt_metrics['accuracy'], dt_metrics['precision'], 
                 dt_metrics['recall'], dt_metrics['f1_score']]
    rf_values = [rf_metrics['accuracy'], rf_metrics['precision'], 
                 rf_metrics['recall'], rf_metrics['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, dt_values, width, label='Decision Tree', color='#E50914')
    bars2 = ax.bar(x + width/2, rf_values, width, label='Random Forest', color='#564d4d')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison\n(Decision Tree vs Random Forest)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def predict_rating(model, rating_encoder, genre_encoder, country_encoder,
                   duration, genre, country, release_year):
    """
    Predict rating for a single movie based on input features.
    """
    try:
        genre_encoded = genre_encoder.transform([genre])[0] if genre in genre_encoder.classes_ else 0
    except:
        genre_encoded = 0
    
    try:
        country_encoded = country_encoder.transform([country])[0] if country in country_encoder.classes_ else 0
    except:
        country_encoded = 0
    
    features = np.array([[duration, genre_encoded, country_encoded, release_year]])
    
    prediction_encoded = model.predict(features)[0]
    prediction = rating_encoder.inverse_transform([prediction_encoded])[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        prob_dict = {
            rating_encoder.inverse_transform([i])[0]: prob 
            for i, prob in enumerate(probabilities)
        }
    else:
        prob_dict = {prediction: 1.0}
    
    return prediction, prob_dict


def create_rating_distribution_plot(df):
    """
    Create a bar plot showing the distribution of ratings in the dataset.
    """
    rating_counts = df['rating_label'].value_counts()
    
    rating_order = ['G', 'PG', 'PG-13', 'R']
    rating_counts = rating_counts.reindex(rating_order)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
    
    bars = ax.bar(rating_counts.index, rating_counts.values, color=colors)
    
    ax.set_xlabel('Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Movie Ratings in Dataset', 
                 fontsize=14, fontweight='bold')
    
    for bar, count in zip(bars, rating_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{int(count):,}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    return fig
