"""
Netflix Global Content Analysis - Streamlit Dashboard
======================================================
Interactive dashboard for analyzing Netflix content trends,
country-wise genre distribution, musical content scoring,
time series analysis, and movie rating prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests

from global_content_analysis import (
    load_and_preprocess_data,
    get_expanded_country_data,
    get_expanded_genre_data,
    get_fully_expanded_data,
    get_country_genre_crosstab,
    get_normalized_genre_distribution,
    create_stacked_bar_chart,
    calculate_musical_content_score,
    get_top_musical_titles,
    create_boxplot_movies_vs_tvshows,
    perform_statistical_test,
    perform_time_series_analysis,
    create_time_series_plot,
    get_trend_analysis_commentary
)

from rating_prediction import (
    prepare_features_for_ml,
    encode_features,
    train_models,
    create_confusion_matrix_plot,
    create_feature_importance_plot,
    create_model_comparison_plot,
    predict_rating,
    create_rating_distribution_plot
)

st.set_page_config(
    page_title="Netflix Data Science Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
    background-color: ;
}

    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: black;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #564d4d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #E50914;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .rating-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .rating-g { background-color: #4CAF50; color: white; }
    .rating-pg { background-color: #FFC107; color: black; }
    .rating-pg13 { background-color: #FF9800; color: white; }
    .rating-r { background-color: #F44336; color: white; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_netflix_data():
    """Load and cache the Netflix dataset."""
    urls = [
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2021/2021-04-20/netflix_titles.csv",
        "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/netflix_titles.csv",
    ]
    
    try:
        df = pd.read_csv('netflix_titles.csv')
        return df
    except FileNotFoundError:
        for url in urls:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                df = pd.read_csv(StringIO(response.text))
                df.to_csv('netflix_titles.csv', index=False)
                return df
            except Exception:
                continue
        st.error("Failed to load dataset from all sources.")
        return None


@st.cache_data
def run_analysis(df):
    """Run all analyses and cache results."""
    df_processed = load_and_preprocess_data(df)
    
    df_country_expanded = get_expanded_country_data(df_processed)
    df_genre_expanded = get_expanded_genre_data(df_processed)
    df_fully_expanded = get_fully_expanded_data(df_processed)
    
    crosstab = get_country_genre_crosstab(df_processed)
    normalized = get_normalized_genre_distribution(crosstab, top_n=10)
    
    df_scored = calculate_musical_content_score(df_processed)
    top_musical = get_top_musical_titles(df_scored, n=10)
    stats_results = perform_statistical_test(df_scored)
    
    monthly_counts = perform_time_series_analysis(df_processed)
    trend_commentary = get_trend_analysis_commentary(monthly_counts)
    
    return {
        'preprocessed': df_processed,
        'country_expanded': df_country_expanded,
        'genre_expanded': df_genre_expanded,
        'fully_expanded': df_fully_expanded,
        'scored': df_scored,
        'crosstab': crosstab,
        'normalized': normalized,
        'top_musical': top_musical,
        'stats': stats_results,
        'time_series': monthly_counts,
        'commentary': trend_commentary
    }


@st.cache_resource
def train_rating_models(df):
    """Train and cache the rating prediction models."""
    df_ml = prepare_features_for_ml(df)
    df_encoded, genre_encoder, country_encoder = encode_features(df_ml)
    model_results = train_models(df_encoded)
    
    return {
        'df_ml': df_ml,
        'df_encoded': df_encoded,
        'genre_encoder': genre_encoder,
        'country_encoder': country_encoder,
        'model_results': model_results
    }


def main():
    st.markdown('<h1 class="main-header">Netflix Global Content Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Data Science Project: Analyzing Content Trends, Musical Influence & Rating Prediction</p>', unsafe_allow_html=True)
    
    with st.spinner("Loading Netflix dataset..."):
        df = load_netflix_data()
    
    if df is None:
        st.error("Unable to load the Netflix dataset. Please check your connection and try again.")
        return
    
    with st.spinner("Running comprehensive analysis..."):
        results = run_analysis(df)
    
    st.sidebar.header("Dataset Overview")
    st.sidebar.metric("Total Titles", f"{len(df):,}")
    st.sidebar.metric("Movies", f"{len(df[df['type'] == 'Movie']):,}")
    st.sidebar.metric("TV Shows", f"{len(df[df['type'] == 'TV Show']):,}")
    
    unique_countries = results['preprocessed']['primary_country'].nunique()
    st.sidebar.metric("Countries", f"{unique_countries}")
    
    st.sidebar.divider()
    st.sidebar.subheader("Required Libraries")
    st.sidebar.code("""
numpy
pandas
matplotlib
seaborn
sklearn
scipy.stats
    """, language="text")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Preprocessing",
        "Country-Genre Analysis", 
        "Musical Content Scoring",
        "Time Series Analysis",
        "Rating Prediction"
    ])
    
    with tab1:
        st.header("Section 1: Data Preprocessing")
        st.markdown("""
        This section handles the initial data preparation:
        - **Missing Values**: Country values filled with 'Unknown'
        - **Country Expansion**: Comma-separated countries split into separate entries (each row = single country)
        - **Primary Country**: First listed country extracted as primary_country feature
        - **Genre Expansion**: listed_in column expanded for single-genre analysis (each row = single genre)
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            missing_country = df['country'].isna().sum()
            st.metric("Missing Countries (Before)", f"{missing_country:,}")
        with col2:
            missing_desc = df['description'].isna().sum()
            st.metric("Missing Descriptions", f"{missing_desc:,}")
        with col3:
            missing_date = df['date_added'].isna().sum()
            st.metric("Missing Dates", f"{missing_date:,}")
        with col4:
            st.metric("Unique Primary Countries", f"{unique_countries}")
        
        st.subheader("Data Expansion Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Rows", f"{len(df):,}")
        with col2:
            st.metric("Country-Expanded Rows", f"{len(results['country_expanded']):,}")
        with col3:
            st.metric("Genre-Expanded Rows", f"{len(results['genre_expanded']):,}")
        
        st.subheader("Sample Preprocessed Data (Before Expansion)")
        sample_cols = ['title', 'type', 'primary_country', 'listed_in', 'date_added']
        st.dataframe(results['preprocessed'][sample_cols].head(10), use_container_width=True)
        
        with st.expander("View Sample Country-Expanded Data (Each row = single country)"):
            country_exp_cols = ['title', 'type', 'country', 'primary_country', 'listed_in']
            st.dataframe(results['country_expanded'][country_exp_cols].head(15), use_container_width=True)
        
        with st.expander("View Sample Genre-Expanded Data (Each row = single genre)"):
            genre_exp_cols = ['title', 'type', 'primary_country', 'listed_in']
            st.dataframe(results['genre_expanded'][genre_exp_cols].head(15), use_container_width=True)
        
        st.subheader("Top 10 Primary Countries by Content Count")
        country_counts = results['preprocessed']['primary_country'].value_counts().head(10)
        
        fig_countries, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(country_counts)))[::-1]
        bars = ax.barh(country_counts.index[::-1], country_counts.values[::-1], color=colors)
        ax.set_xlabel('Number of Titles')
        ax.set_title('Top 10 Content-Producing Countries')
        for bar, value in zip(bars, country_counts.values[::-1]):
            ax.text(value + 10, bar.get_y() + bar.get_height()/2, f'{value:,}', 
                   va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_countries)
        plt.close()
    
    with tab2:
        st.header("Section 2: Country-Wise Genre Dominance")
        st.markdown("""
        **Strategic Analysis 1**: Understanding how different countries focus on specific genres
        reveals distinct content strategies in the global entertainment market.
        """)
        
        st.subheader("Cross-Tabulation: Country vs Genre")
        st.markdown("Showing counts of content for each country-genre pair (Top 10 countries, Top 8 genres):")
        
        crosstab = results['crosstab']
        top_10_countries = crosstab.sum(axis=1).nlargest(10).index
        top_8_genres = crosstab.sum(axis=0).nlargest(8).index
        display_crosstab = crosstab.loc[top_10_countries, top_8_genres]
        st.dataframe(display_crosstab, use_container_width=True)
        
        st.subheader("Normalized Genre Distribution (%)")
        st.markdown("Percentage of each genre within the top content-producing countries:")
        
        normalized = results['normalized']
        top_8_genre_cols = normalized.sum().nlargest(8).index.tolist()
        normalized_display = normalized[top_8_genre_cols]
        st.dataframe(normalized_display.round(2), use_container_width=True)
        
        st.subheader("Genre Distribution Strategy: Top 5 Countries")
        st.markdown("""
        This visualization reveals the distinct content strategies of leading countries:
        - **United States**: Diverse portfolio with emphasis on Comedies and Documentaries
        - **India**: Strong focus on International Movies and Dramas
        - **United Kingdom**: Balanced mix with British TV Shows prominence
        """)
        
        fig_stacked = create_stacked_bar_chart(normalized, top_n=5)
        st.pyplot(fig_stacked)
        plt.close()
    
    with tab3:
        st.header("Section 3: Musical Content Scoring")
        st.markdown("""
        **Creative Feature Engineering**: Quantifying the potential for titles to have 
        significant musical content using NLP techniques and genre weighting.
        """)
        
        st.subheader("Feature Engineering Methodology")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Musical Keywords (CountVectorizer)**")
            st.code("""
keywords = ['musical', 'song', 'sing', 
            'soundtrack', 'album', 
            'concert', 'music']
            """)
            
        with col2:
            st.markdown("**Score Formula**")
            st.latex(r"\text{Score} = \text{keyword\_count} + (\text{is\_musical\_genre} \times 5)")
        
        st.subheader("Top 10 Titles: Highest Musical Content Score")
        top_musical = results['top_musical'].copy()
        display_cols = ['title', 'type', 'Musical_Content_Score', 'musical_keyword_count', 'is_musical_genre']
        st.dataframe(top_musical[display_cols], use_container_width=True)
        
        with st.expander("View Descriptions of Top Musical Titles"):
            for idx, row in top_musical.iterrows():
                st.markdown(f"**{row['title']}** (Score: {row['Musical_Content_Score']})")
                st.markdown(f"*{row['description'][:300]}...*" if len(str(row['description'])) > 300 else f"*{row['description']}*")
                st.divider()
        
        st.subheader("Statistical Analysis: Movies vs TV Shows")
        
        fig_box = create_boxplot_movies_vs_tvshows(results['scored'])
        st.pyplot(fig_box)
        plt.close()
        
        stats = results['stats']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Mann-Whitney U Test**")
            st.metric("U-Statistic", f"{stats['Mann-Whitney U']['statistic']:,.2f}")
            st.metric("P-Value", f"{stats['Mann-Whitney U']['p_value']:.6f}")
            if stats['Mann-Whitney U']['p_value'] < 0.05:
                st.success("Significant difference detected (p < 0.05)")
            else:
                st.info("No significant difference (p >= 0.05)")
        
        with col2:
            st.markdown("**Independent T-Test**")
            st.metric("T-Statistic", f"{stats['T-Test']['statistic']:.4f}")
            st.metric("P-Value", f"{stats['T-Test']['p_value']:.6f}")
            if stats['T-Test']['p_value'] < 0.05:
                st.success("Significant difference detected (p < 0.05)")
            else:
                st.info("No significant difference (p >= 0.05)")
        
        st.subheader("Descriptive Statistics")
        desc = stats['descriptive_stats']
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Movies**")
            st.write(f"- Mean Score: {desc['movies_mean']:.4f}")
            st.write(f"- Std Deviation: {desc['movies_std']:.4f}")
            st.write(f"- Count: {desc['movies_count']:,}")
        with col2:
            st.markdown("**TV Shows**")
            st.write(f"- Mean Score: {desc['tvshows_mean']:.4f}")
            st.write(f"- Std Deviation: {desc['tvshows_std']:.4f}")
            st.write(f"- Count: {desc['tvshows_count']:,}")
    
    with tab4:
        st.header("Section 4: Time Series Analysis")
        st.markdown("""
        **Temporal Analysis**: Examining the pattern of content additions over time
        using monthly aggregation and 6-month rolling averages to identify trends.
        """)
        
        time_data = results['time_series']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Months Analyzed", len(time_data))
        with col2:
            st.metric("Avg Monthly Additions", f"{time_data['count'].mean():.1f}")
        with col3:
            peak = time_data.loc[time_data['count'].idxmax()]
            st.metric("Peak Month", peak['date'].strftime('%b %Y'))
        with col4:
            st.metric("Peak Additions", f"{int(peak['count']):,}")
        
        st.subheader("Monthly Content Additions with 6-Month Rolling Mean")
        
        fig_time = create_time_series_plot(time_data)
        st.pyplot(fig_time)
        plt.close()
        
        st.subheader("Strategic Implications")
        st.markdown(results['commentary'])
        
        st.subheader("Monthly Data Table")
        display_time = time_data.copy()
        display_time['date'] = display_time['date'].dt.strftime('%Y-%m')
        display_time.columns = ['Year-Month', 'New Titles', 'Date', '6-Month Rolling Avg']
        st.dataframe(display_time[['Date', 'New Titles', '6-Month Rolling Avg']].tail(24), 
                    use_container_width=True)
    
    with tab5:
        st.header("Section 5: Movie Rating Prediction")
        st.markdown("""
        **Machine Learning Classification**: Predict movie ratings (G, PG, PG-13, R) using 
        Decision Tree and Random Forest models based on content features.
        """)
        
        st.subheader("Feature Engineering for ML")
        st.markdown("""
        The following features are used to predict movie ratings:
        - **Duration**: Movie length in minutes
        - **Genre**: Primary genre category (encoded)
        - **Country**: Primary production country (encoded)
        - **Release Year**: Year the content was released
        """)
        
        with st.spinner("Training ML models..."):
            ml_data = train_rating_models(df)
        
        df_ml = ml_data['df_ml']
        model_results = ml_data['model_results']
        genre_encoder = ml_data['genre_encoder']
        country_encoder = ml_data['country_encoder']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Training Samples", f"{len(df_ml):,}")
        with col2:
            st.metric("Features Used", "4")
        with col3:
            dt_acc = model_results['decision_tree']['metrics']['accuracy']
            st.metric("Decision Tree Accuracy", f"{dt_acc:.1%}")
        with col4:
            rf_acc = model_results['random_forest']['metrics']['accuracy']
            st.metric("Random Forest Accuracy", f"{rf_acc:.1%}")
        
        st.subheader("Rating Distribution in Dataset")
        fig_dist = create_rating_distribution_plot(df_ml)
        st.pyplot(fig_dist)
        plt.close()
        
        st.subheader("Model Performance Comparison")
        fig_compare = create_model_comparison_plot(
            model_results['decision_tree']['metrics'],
            model_results['random_forest']['metrics']
        )
        st.pyplot(fig_compare)
        plt.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Decision Tree Results")
            dt_metrics = model_results['decision_tree']['metrics']
            st.write(f"- **Accuracy**: {dt_metrics['accuracy']:.4f}")
            st.write(f"- **Precision**: {dt_metrics['precision']:.4f}")
            st.write(f"- **Recall**: {dt_metrics['recall']:.4f}")
            st.write(f"- **F1-Score**: {dt_metrics['f1_score']:.4f}")
            
            st.markdown("**Confusion Matrix**")
            fig_dt_cm = create_confusion_matrix_plot(
                dt_metrics['confusion_matrix'],
                dt_metrics['classes'],
                "Decision Tree Confusion Matrix"
            )
            st.pyplot(fig_dt_cm)
            plt.close()
        
        with col2:
            st.markdown("### Random Forest Results")
            rf_metrics = model_results['random_forest']['metrics']
            st.write(f"- **Accuracy**: {rf_metrics['accuracy']:.4f}")
            st.write(f"- **Precision**: {rf_metrics['precision']:.4f}")
            st.write(f"- **Recall**: {rf_metrics['recall']:.4f}")
            st.write(f"- **F1-Score**: {rf_metrics['f1_score']:.4f}")
            
            st.markdown("**Confusion Matrix**")
            fig_rf_cm = create_confusion_matrix_plot(
                rf_metrics['confusion_matrix'],
                rf_metrics['classes'],
                "Random Forest Confusion Matrix"
            )
            st.pyplot(fig_rf_cm)
            plt.close()
        
        st.subheader("Feature Importance Analysis")
        col1, col2 = st.columns(2)
        
        feature_names = ['Duration (min)', 'Genre', 'Country', 'Release Year']
        
        with col1:
            fig_dt_imp = create_feature_importance_plot(
                model_results['decision_tree']['model'],
                feature_names,
                "Decision Tree Feature Importance"
            )
            st.pyplot(fig_dt_imp)
            plt.close()
        
        with col2:
            fig_rf_imp = create_feature_importance_plot(
                model_results['random_forest']['model'],
                feature_names,
                "Random Forest Feature Importance"
            )
            st.pyplot(fig_rf_imp)
            plt.close()
        
        st.divider()
        st.subheader("Interactive Rating Prediction")
        st.markdown("Enter movie details below to predict its rating:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            duration_input = st.slider("Duration (minutes)", 30, 240, 90)
            
            available_genres = sorted(genre_encoder.classes_.tolist())
            genre_input = st.selectbox("Primary Genre", available_genres, index=0)
        
        with col2:
            available_countries = sorted(country_encoder.classes_.tolist())
            country_input = st.selectbox("Country", available_countries, 
                                         index=available_countries.index("United States") if "United States" in available_countries else 0)
            
            year_input = st.slider("Release Year", 1950, 2025, 2020)
        
        model_choice = st.radio("Select Model", ["Random Forest (Recommended)", "Decision Tree"], horizontal=True)
        
        if st.button("Predict Rating", type="primary"):
            selected_model = model_results['random_forest']['model'] if "Random Forest" in model_choice else model_results['decision_tree']['model']
            
            prediction, probabilities = predict_rating(
                selected_model,
                model_results['rating_encoder'],
                genre_encoder,
                country_encoder,
                duration_input,
                genre_input,
                country_input,
                year_input
            )
            
            rating_colors = {'G': 'rating-g', 'PG': 'rating-pg', 'PG-13': 'rating-pg13', 'R': 'rating-r'}
            rating_class = rating_colors.get(prediction, 'rating-pg')
            
            st.markdown(f"""
            <div class="rating-box {rating_class}">
                Predicted Rating: {prediction}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Prediction Probabilities:**")
            prob_df = pd.DataFrame({
                'Rating': list(probabilities.keys()),
                'Probability': list(probabilities.values())
            }).sort_values('Probability', ascending=False)
            
            for _, row in prob_df.iterrows():
                st.progress(row['Probability'], text=f"{row['Rating']}: {row['Probability']:.1%}")
        
        with st.expander("Model Details & Methodology"):
            st.markdown("""
            ### About the Rating Prediction Models
            
            This model is trained on **real MPAA ratings** from the Netflix dataset. 
            The dataset is filtered to include only content with standard MPAA ratings:
            G, PG, PG-13, and R.
            
            ### Models Used
            
            1. **Decision Tree Classifier**
               - Max Depth: 10
               - Min Samples Split: 10
               - Interpretable and fast
            
            2. **Random Forest Classifier**
               - 100 Estimators
               - Max Depth: 15
               - Generally more accurate due to ensemble learning
            
            ### Feature Encoding
            
            - **Genre & Country**: Label encoded to numeric values
            - **Duration**: Movie length in minutes (numeric)
            - **Release Year**: Year of release (numeric)
            
            ### Note on Predictions
            
            If your input falls outside categories seen during training, the model 
            will use the closest available encoding. Results are based on patterns 
            learned from actual Netflix content ratings.
            """)
    
    st.divider()
    st.markdown("""
    ---
    **Netflix Global Content Analysis** | Built with Streamlit, Pandas, Matplotlib, Seaborn, Scikit-learn & SciPy
    
    *Data Source: Netflix Movies and TV Shows Dataset*
    """)


if __name__ == "__main__":
    main()
