"""
Netflix Global Content Analysis
===============================
This module performs comprehensive data science analysis on Netflix content data,
including country-wise genre dominance, musical content scoring, and time series analysis.

Required Libraries: numpy, pandas, matplotlib, seaborn, sklearn, scipy.stats
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from scipy import stats
import io
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_and_preprocess_data(df):
    """
    Section 1: Data Preprocessing and Country/Genre Expansion
    ----------------------------------------------------------
    - Handle missing country values by filling with 'Unknown'
    - Create primary_country feature taking the first country listed (before expansion)
    - Expand country column so each entry represents a single country
    - Expand listed_in column so each row has a single genre
    
    Returns both the base preprocessed dataframe and expanded versions.
    """
    df_processed = df.copy()
    
    df_processed['country'] = df_processed['country'].fillna('Unknown')
    df_processed['description'] = df_processed['description'].fillna('')
    df_processed['listed_in'] = df_processed['listed_in'].fillna('Unknown')
    df_processed['date_added'] = pd.to_datetime(df_processed['date_added'], errors='coerce')
    
    df_processed['primary_country'] = df_processed['country'].apply(
        lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown'
    )
    
    df_processed['original_listed_in'] = df_processed['listed_in']
    
    return df_processed


def get_expanded_country_data(df):
    """
    Expand the country column so each entry represents a single country.
    Handles comma-separated values. Returns fully expanded dataframe.
    """
    df_expanded = df.copy()
    df_expanded['country'] = df_expanded['country'].str.split(',')
    df_expanded = df_expanded.explode('country')
    df_expanded['country'] = df_expanded['country'].str.strip()
    return df_expanded


def get_expanded_genre_data(df):
    """
    Expand the listed_in column so each row has a single genre.
    Handles comma-separated values. Returns fully expanded dataframe.
    """
    df_expanded = df.copy()
    df_expanded['listed_in'] = df_expanded['listed_in'].str.split(',')
    df_expanded = df_expanded.explode('listed_in')
    df_expanded['listed_in'] = df_expanded['listed_in'].str.strip()
    return df_expanded


def get_fully_expanded_data(df):
    """
    Expand both country and listed_in columns so each row has a single country
    and a single genre. This is the fully expanded dataset as per specification.
    """
    df_expanded = df.copy()
    
    df_expanded['country'] = df_expanded['country'].str.split(',')
    df_expanded = df_expanded.explode('country')
    df_expanded['country'] = df_expanded['country'].str.strip()
    
    df_expanded['listed_in'] = df_expanded['listed_in'].str.split(',')
    df_expanded = df_expanded.explode('listed_in')
    df_expanded['listed_in'] = df_expanded['listed_in'].str.strip()
    
    return df_expanded


def get_country_genre_crosstab(df):
    """
    Section 2: Strategic Analysis 1 - Country-Wise Genre Dominance
    ---------------------------------------------------------------
    Create cross-tabulation (pivot table) of primary_country and listed_in,
    showing the count of content for each country-genre pair.
    Uses the genre-expanded dataframe for accurate counts.
    """
    df_expanded = get_expanded_genre_data(df)
    crosstab = pd.crosstab(df_expanded['primary_country'], df_expanded['listed_in'])
    return crosstab


def get_normalized_genre_distribution(crosstab, top_n=10):
    """
    Normalize the cross-tabulation table to show percentage distribution
    of genres within each of the top N content-producing countries.
    """
    country_totals = crosstab.sum(axis=1)
    top_countries = country_totals.nlargest(top_n).index
    
    top_crosstab = crosstab.loc[top_countries]
    
    normalized = top_crosstab.div(top_crosstab.sum(axis=1), axis=0) * 100
    
    return normalized


def create_stacked_bar_chart(normalized_df, top_n=5):
    """
    Generate a stacked bar chart for Top N Countries showing internal genre distribution.
    This visualization communicates distinct content strategies of each country.
    """
    country_totals = normalized_df.sum(axis=1)
    top_countries = country_totals.nlargest(top_n).index
    plot_data = normalized_df.loc[top_countries]
    
    genre_totals = plot_data.sum(axis=0)
    top_genres = genre_totals.nlargest(8).index
    plot_data_filtered = plot_data[top_genres]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_genres)))
    
    bottom = np.zeros(len(plot_data_filtered))
    
    for i, genre in enumerate(top_genres):
        values = plot_data_filtered[genre].values
        ax.barh(plot_data_filtered.index, values, left=bottom, label=genre, color=colors[i])
        bottom += values
    
    ax.set_xlabel('Percentage of Content (%)', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    ax.set_title('Top 5 Countries: Genre Distribution Strategy\n(Showing distinct content strategies by country)', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Genre', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    return fig


def calculate_musical_keyword_count(df):
    """
    Section 3: Creative Feature Engineering - Musical Content Score
    ----------------------------------------------------------------
    Use CountVectorizer to count occurrences of musical keywords in descriptions.
    Keywords: ['musical', 'song', 'sing', 'soundtrack', 'album', 'concert', 'music']
    """
    musical_keywords = ['musical', 'song', 'sing', 'soundtrack', 'album', 'concert', 'music']
    
    vectorizer = CountVectorizer(vocabulary=musical_keywords, lowercase=True)
    
    descriptions = df['description'].fillna('').astype(str)
    
    keyword_matrix = vectorizer.fit_transform(descriptions)
    
    keyword_counts = np.array(keyword_matrix.sum(axis=1)).flatten()
    
    return keyword_counts


def create_musical_genre_flag(df):
    """
    Create a binary flag is_musical_genre that is 1 if the title belongs
    to Music or Concert genres (based on original listed_in column).
    Uses 'original_listed_in' if available to check full genre list.
    """
    musical_genres = ['music', 'concert']
    
    genre_column = 'original_listed_in' if 'original_listed_in' in df.columns else 'listed_in'
    
    is_musical = df[genre_column].fillna('').str.lower().apply(
        lambda x: 1 if any(genre in x for genre in musical_genres) else 0
    )
    
    return is_musical


def calculate_musical_content_score(df):
    """
    Calculate the final Musical_Content_Score combining features:
    Score = musical_keyword_count + (is_musical_genre * 5)
    """
    df_scored = df.copy()
    
    df_scored['musical_keyword_count'] = calculate_musical_keyword_count(df)
    df_scored['is_musical_genre'] = create_musical_genre_flag(df)
    df_scored['Musical_Content_Score'] = (
        df_scored['musical_keyword_count'] + (df_scored['is_musical_genre'] * 5)
    )
    
    return df_scored


def get_top_musical_titles(df, n=10):
    """
    Section 4: Strategic Analysis 2 - High-Score Music Content
    -----------------------------------------------------------
    Identify and return the Top N Titles with highest Musical_Content_Score.
    """
    top_titles = df.nlargest(n, 'Musical_Content_Score')[
        ['title', 'type', 'listed_in', 'Musical_Content_Score', 
         'musical_keyword_count', 'is_musical_genre', 'description']
    ]
    return top_titles


def create_boxplot_movies_vs_tvshows(df):
    """
    Create a box plot comparing Musical_Content_Score of Movies versus TV Shows.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'Movie': '#E50914', 'TV Show': '#564d4d'}
    
    sns.boxplot(
        data=df, 
        x='type', 
        y='Musical_Content_Score',
        palette=colors,
        ax=ax
    )
    
    ax.set_xlabel('Content Type', fontsize=12)
    ax.set_ylabel('Musical Content Score', fontsize=12)
    ax.set_title('Musical Content Score: Movies vs TV Shows\n(Comparing potential for musical content)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def perform_statistical_test(df):
    """
    Perform Mann-Whitney U and T-test to determine if mean Musical_Content_Score
    is significantly different between Movies and TV Shows.
    Returns test statistics, p-values, and conclusions.
    """
    movies_scores = df[df['type'] == 'Movie']['Musical_Content_Score'].dropna()
    tvshows_scores = df[df['type'] == 'TV Show']['Musical_Content_Score'].dropna()
    
    mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(
        movies_scores, tvshows_scores, alternative='two-sided'
    )
    
    ttest_stat, ttest_p = stats.ttest_ind(movies_scores, tvshows_scores)
    
    alpha = 0.05
    
    if mannwhitney_p < alpha:
        mw_conclusion = f"Significant difference (p={mannwhitney_p:.6f} < {alpha})"
    else:
        mw_conclusion = f"No significant difference (p={mannwhitney_p:.6f} >= {alpha})"
    
    if ttest_p < alpha:
        tt_conclusion = f"Significant difference (p={ttest_p:.6f} < {alpha})"
    else:
        tt_conclusion = f"No significant difference (p={ttest_p:.6f} >= {alpha})"
    
    results = {
        'Mann-Whitney U': {
            'statistic': mannwhitney_stat,
            'p_value': mannwhitney_p,
            'conclusion': mw_conclusion
        },
        'T-Test': {
            'statistic': ttest_stat,
            'p_value': ttest_p,
            'conclusion': tt_conclusion
        },
        'descriptive_stats': {
            'movies_mean': movies_scores.mean(),
            'movies_std': movies_scores.std(),
            'movies_count': len(movies_scores),
            'tvshows_mean': tvshows_scores.mean(),
            'tvshows_std': tvshows_scores.std(),
            'tvshows_count': len(tvshows_scores)
        }
    }
    
    return results


def perform_time_series_analysis(df):
    """
    Section 5: Time Series Analysis on date_added column
    -----------------------------------------------------
    - Aggregate data to show monthly count of new titles
    - Calculate 6-month rolling mean using Pandas
    - Prepare data for visualization
    """
    df_time = df.dropna(subset=['date_added']).copy()
    
    df_time['year_month'] = df_time['date_added'].dt.to_period('M')
    
    monthly_counts = df_time.groupby('year_month').size().reset_index(name='count')
    
    monthly_counts['date'] = monthly_counts['year_month'].dt.to_timestamp()
    
    monthly_counts = monthly_counts.sort_values('date')
    
    monthly_counts['rolling_mean_6m'] = monthly_counts['count'].rolling(window=6, min_periods=1).mean()
    
    return monthly_counts


def create_time_series_plot(monthly_counts):
    """
    Visualize raw monthly counts and 6-month rolling mean trend using Matplotlib.
    Include commentary on strategic implications of the trend.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.bar(monthly_counts['date'], monthly_counts['count'], 
           alpha=0.5, color='#E50914', label='Monthly New Titles', width=25)
    
    ax.plot(monthly_counts['date'], monthly_counts['rolling_mean_6m'], 
            color='#221f1f', linewidth=3, label='6-Month Rolling Mean', marker='o', markersize=3)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Titles Added', fontsize=12)
    ax.set_title('Netflix Content Addition Trend Over Time\n(Monthly counts with 6-month rolling average)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    ax.tick_params(axis='x', rotation=45)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def get_trend_analysis_commentary(monthly_counts):
    """
    Generate strategic implications commentary based on the time series trend.
    """
    if len(monthly_counts) < 2:
        return "Insufficient data for trend analysis."
    
    mid_point = len(monthly_counts) // 2
    first_half_avg = monthly_counts['rolling_mean_6m'].iloc[:mid_point].mean()
    second_half_avg = monthly_counts['rolling_mean_6m'].iloc[mid_point:].mean()
    
    peak_month = monthly_counts.loc[monthly_counts['count'].idxmax()]
    lowest_month = monthly_counts.loc[monthly_counts['count'].idxmin()]
    
    overall_trend = (second_half_avg - first_half_avg) / first_half_avg * 100 if first_half_avg > 0 else 0
    
    recent_data = monthly_counts.tail(12)
    recent_trend = recent_data['rolling_mean_6m'].iloc[-1] - recent_data['rolling_mean_6m'].iloc[0]
    
    commentary = []
    
    commentary.append("### Strategic Implications of Netflix Content Trend")
    commentary.append("")
    
    if overall_trend > 20:
        commentary.append(f"**Growth Phase Identified:** The overall trend shows a {overall_trend:.1f}% increase in content additions, "
                         "indicating Netflix's aggressive expansion strategy to capture market share globally.")
    elif overall_trend < -20:
        commentary.append(f"**Consolidation Phase:** The {abs(overall_trend):.1f}% decrease in content additions suggests "
                         "Netflix is focusing on content quality over quantity, possibly in response to increased competition.")
    else:
        commentary.append(f"**Stable Growth Pattern:** With a {abs(overall_trend):.1f}% change, Netflix maintains "
                         "a consistent content acquisition strategy.")
    
    commentary.append("")
    commentary.append(f"**Peak Activity:** {peak_month['date'].strftime('%B %Y')} saw the highest content addition "
                     f"({int(peak_month['count'])} titles), possibly aligned with holiday season or strategic launches.")
    
    commentary.append("")
    commentary.append(f"**Minimum Activity:** {lowest_month['date'].strftime('%B %Y')} had the lowest additions "
                     f"({int(lowest_month['count'])} titles), which could indicate licensing cycles or seasonal patterns.")
    
    commentary.append("")
    if recent_trend > 0:
        commentary.append("**Recent Momentum:** The 6-month rolling average shows positive momentum in recent periods, "
                         "suggesting continued investment in content library expansion.")
    else:
        commentary.append("**Recent Trend:** The 6-month rolling average indicates a slowdown in recent periods, "
                         "which may reflect market maturation or strategic shift towards original content production.")
    
    commentary.append("")
    commentary.append("**Strategic Recommendations:**")
    commentary.append("1. Monitor seasonal patterns to optimize content release timing")
    commentary.append("2. Analyze correlation between content additions and subscriber growth")
    commentary.append("3. Compare regional content strategies based on these temporal patterns")
    
    return "\n".join(commentary)


def run_complete_analysis(df):
    """
    Run the complete Netflix content analysis pipeline.
    Returns all results and figures for display.
    """
    results = {}
    
    print("=" * 60)
    print("NETFLIX GLOBAL CONTENT ANALYSIS")
    print("=" * 60)
    
    print("\n[1/5] Preprocessing data...")
    df_processed = load_and_preprocess_data(df)
    results['preprocessed_data'] = df_processed
    print(f"    Loaded {len(df_processed)} titles")
    
    print("\n[2/5] Analyzing country-genre dominance...")
    crosstab = get_country_genre_crosstab(df_processed)
    normalized = get_normalized_genre_distribution(crosstab, top_n=10)
    results['crosstab'] = crosstab
    results['normalized_distribution'] = normalized
    results['country_genre_chart'] = create_stacked_bar_chart(normalized, top_n=5)
    print(f"    Created cross-tabulation for {len(crosstab)} countries and {len(crosstab.columns)} genres")
    
    print("\n[3/5] Calculating musical content scores...")
    df_scored = calculate_musical_content_score(df_processed)
    results['scored_data'] = df_scored
    results['top_musical_titles'] = get_top_musical_titles(df_scored, n=10)
    print(f"    Calculated scores for {len(df_scored)} titles")
    
    print("\n[4/5] Performing statistical analysis...")
    results['boxplot'] = create_boxplot_movies_vs_tvshows(df_scored)
    results['statistical_tests'] = perform_statistical_test(df_scored)
    print(f"    Completed Mann-Whitney U and T-test analyses")
    
    print("\n[5/5] Analyzing time series trends...")
    monthly_counts = perform_time_series_analysis(df_processed)
    results['time_series_data'] = monthly_counts
    results['time_series_plot'] = create_time_series_plot(monthly_counts)
    results['trend_commentary'] = get_trend_analysis_commentary(monthly_counts)
    print(f"    Analyzed {len(monthly_counts)} months of content additions")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    print("Loading Netflix dataset...")
    try:
        df = pd.read_csv('netflix_titles.csv')
        results = run_complete_analysis(df)
        
        print("\n\nTOP 10 TITLES WITH HIGHEST MUSICAL CONTENT SCORE:")
        print("-" * 50)
        print(results['top_musical_titles'][['title', 'Musical_Content_Score', 'type']].to_string())
        
        print("\n\nSTATISTICAL TEST RESULTS:")
        print("-" * 50)
        stats_results = results['statistical_tests']
        print(f"Mann-Whitney U Test: {stats_results['Mann-Whitney U']['conclusion']}")
        print(f"T-Test: {stats_results['T-Test']['conclusion']}")
        
        print("\n\nTIME SERIES TREND COMMENTARY:")
        print("-" * 50)
        print(results['trend_commentary'])
        
        plt.show()
        
    except FileNotFoundError:
        print("Error: netflix_titles.csv not found. Please ensure the dataset is in the current directory.")
