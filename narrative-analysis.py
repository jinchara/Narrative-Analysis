# NARRATIVE AND COORDINATION ANALYSIS FOR GEORGIAN DISINFORMATION RESEARCH
# Coordination Activation Index (CAI) Implementation
# Author: ForSet Advanced Fellowship Project
# Purpose: Detect coordinated posts and analyze temporal patterns

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
import warnings
warnings.filterwarnings('ignore')

class CoordinationAnalyzer:
    """
    Analyzes coordinated disinformation operations using the Coordination Activation Index (CAI)
    """
    
    def __init__(self, 
                 similarity_threshold=0.85,
                 time_window_minutes=60,
                 min_word_length=10):
        """
        Initialize the Coordination Analyzer
        
        Args:
            similarity_threshold: Minimum cosine similarity for coordination (0.85 = 85%)
            time_window_minutes: Maximum time difference for coordination (default: 30 minutes)
            min_word_length: Minimum words in post to analyze
        """
        self.similarity_threshold = similarity_threshold
        self.time_window = timedelta(minutes=time_window_minutes)
        self.min_word_length = min_word_length
        self.vectorizer = None
        
    def load_and_prepare_data(self, excel_file):
        """Load data from Meta Content Library export"""
        print(f" Loading data from: {excel_file}")
        
        try:
            df = pd.read_excel(excel_file)
            print(f" Successfully loaded {len(df)} posts")
        except Exception as e:
            raise Exception(f" Could not read Excel file: {str(e)}")
        
        # Validate required columns
        required_columns = ['id', 'post_owner.name', 'text', 'creation_time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise Exception(f" Missing required columns: {missing_columns}")
        
        # Convert creation_time to datetime
        df['creation_time'] = pd.to_datetime(df['creation_time'], utc=True)
        
        # Clean text content
        df['clean_text'] = df['text'].fillna('').astype(str).str.strip()
        df['clean_text'] = df['clean_text'].str.replace(r'<[^>]+>', '', regex=True)
        df['clean_text'] = df['clean_text'].str.replace(r'\s+', ' ', regex=True)
        df['word_count'] = df['clean_text'].str.split().str.len()
        
        # Filter out short posts
        df = df[df['word_count'] >= self.min_word_length].copy()
        
        # Sort by time
        df = df.sort_values('creation_time').reset_index(drop=True)
        
        print(f" After filtering: {len(df)} posts from {df['post_owner.name'].nunique()} pages")
        print(f" Date range: {df['creation_time'].min()} to {df['creation_time'].max()}")
        
        return df
    
    def detect_coordinated_posts(self, df):
        """
        Detect coordinated post pairs using content similarity + temporal proximity
        
        Returns:
            DataFrame with coordinated post pairs
        """
        print("\n STEP 1: DETECTING COORDINATION")
        print("=" * 60)
        
        # Create TF-IDF vectors
        print(" Creating TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=1,
            ngram_range=(1, 3),
            sublinear_tf=True
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(df['clean_text'])
        print(f" TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Calculate cosine similarity
        print(" Computing cosine similarity...")
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find coordinated pairs
        print(f" Finding coordinated pairs (similarity >= {self.similarity_threshold}, time <= {self.time_window.total_seconds()/60} min)...")
        
        coordinated_pairs = []
        
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                # Check content similarity
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    # Check temporal proximity
                    time_diff = abs((df.iloc[j]['creation_time'] - df.iloc[i]['creation_time']).total_seconds() / 60)
                    
                    if time_diff <= self.time_window.total_seconds() / 60:
                        # Different pages only
                        if df.iloc[i]['post_owner.name'] != df.iloc[j]['post_owner.name']:
                            coordinated_pairs.append({
                                'post_id_1': df.iloc[i]['id'],
                                'post_id_2': df.iloc[j]['id'],
                                'page_1': df.iloc[i]['post_owner.name'],
                                'page_2': df.iloc[j]['post_owner.name'],
                                'time_1': df.iloc[i]['creation_time'],
                                'time_2': df.iloc[j]['creation_time'],
                                'time_diff_minutes': round(time_diff, 2),
                                'similarity': round(similarity_matrix[i, j], 3),
                                'text_1': df.iloc[i]['clean_text'][:200] + "...",
                                'text_2': df.iloc[j]['clean_text'][:200] + "..."
                            })
        
        coord_df = pd.DataFrame(coordinated_pairs)
        
        if len(coord_df) > 0:
            print(f" Found {len(coord_df)} coordinated post pairs!")
            print(f" Involving {len(set(coord_df['page_1'].tolist() + coord_df['page_2'].tolist()))} pages")
        else:
            print(" No coordinated pairs found. Try lowering thresholds.")
        
        return coord_df
    
    def calculate_cai(self, coord_df, df):
        """
        Calculate Coordination Activation Index (CAI) per day
        
        CAI = Coordination Score × Network Coverage × Speed Index
        """
        print("\n STEP 2: CALCULATING COORDINATION ACTIVATION INDEX (CAI)")
        print("=" * 60)
        
        if len(coord_df) == 0:
            print(" No coordinated posts to calculate CAI")
            return pd.DataFrame()
        
        # Add date column
        coord_df['date'] = coord_df['time_1'].dt.date
        
        # Total pages in network
        total_pages = df['post_owner.name'].nunique()
        
        cai_results = []
        
        for date in coord_df['date'].unique():
            daily_coord = coord_df[coord_df['date'] == date]
            
            # Coordination Score: raw count of coordinated posts
            coordination_score = len(daily_coord)
            
            # Network Coverage: unique pages involved / total pages
            unique_pages = set(daily_coord['page_1'].tolist() + daily_coord['page_2'].tolist())
            network_coverage = len(unique_pages) / total_pages
            
            # Speed Index: 1 / average time interval
            avg_time_interval = daily_coord['time_diff_minutes'].mean()
            speed_index = 1 / avg_time_interval if avg_time_interval > 0 else 0
            
            # CAI calculation
            cai = coordination_score * network_coverage * speed_index
            
            cai_results.append({
                'date': date,
                'coordination_score': coordination_score,
                'unique_pages': len(unique_pages),
                'network_coverage': round(network_coverage, 3),
                'avg_time_interval_min': round(avg_time_interval, 2),
                'speed_index': round(speed_index, 4),
                'CAI': round(cai, 4)
            })
        
        cai_df = pd.DataFrame(cai_results).sort_values('date')
        
        print(f" Calculated CAI for {len(cai_df)} days")
        print("\n CAI Summary Statistics:")
        print(cai_df[['CAI', 'coordination_score', 'network_coverage', 'speed_index']].describe())
        
        return cai_df
    
    def identify_network_leaders(self, coord_df):
        """
        Identify seeder pages and hubs using network analysis
        """
        print("\n STEP 3: IDENTIFYING NETWORK LEADERS (SEEDERS & HUBS)")
        print("=" * 60)
        
        if len(coord_df) == 0:
            print(" No coordinated posts to analyze network")
            return None, None
        
        # Create network graph
        G = nx.Graph()
        
        # Add edges weighted by coordination frequency
        for _, row in coord_df.iterrows():
            if G.has_edge(row['page_1'], row['page_2']):
                G[row['page_1']][row['page_2']]['weight'] += 1
            else:
                G.add_edge(row['page_1'], row['page_2'], weight=1)
        
        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(G, weight='weight')
        degree = dict(G.degree(weight='weight'))
        
        # Identify seeders (pages that post first most often)
        seeder_scores = defaultdict(lambda: {'first_count': 0, 'avg_time_lag': []})
        
        for _, row in coord_df.iterrows():
            # Earlier post is the "seeder"
            if row['time_1'] <= row['time_2']:
                seeder = row['page_1']
                time_lag = row['time_diff_minutes']
            else:
                seeder = row['page_2']
                time_lag = row['time_diff_minutes']
            
            seeder_scores[seeder]['first_count'] += 1
            seeder_scores[seeder]['avg_time_lag'].append(time_lag)
        
        # Create leader dataframe
        leaders = []
        for page in G.nodes():
            avg_lag = np.mean(seeder_scores[page]['avg_time_lag']) if seeder_scores[page]['avg_time_lag'] else 0
            
            leaders.append({
                'page': page,
                'betweenness_centrality': round(betweenness.get(page, 0), 4),
                'weighted_degree': degree.get(page, 0),
                'seeder_count': seeder_scores[page]['first_count'],
                'avg_time_lag_min': round(avg_lag, 2)
            })
        
        leaders_df = pd.DataFrame(leaders).sort_values('betweenness_centrality', ascending=False)
        
        print(f" Identified {len(leaders_df)} pages in coordination network")
        print("\n Top 10 Network Leaders (by Betweenness Centrality):")
        print(leaders_df.head(10).to_string(index=False))
        
        return leaders_df, G
    
    def visualize_cai_timeline(self, cai_df, output_file='cai_timeline.png'):
        """Create CAI time series visualization"""
        print(f"\n Creating CAI timeline visualization...")
        
        if len(cai_df) == 0:
            print(" No CAI data to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(cai_df['date'], cai_df['CAI'], marker='o', linewidth=2, markersize=8, color='#e74c3c')
        ax.fill_between(cai_df['date'], cai_df['CAI'], alpha=0.3, color='#e74c3c')
        
        ax.set_title('Coordination Activation Index (CAI) Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('CAI Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f" CAI timeline saved: {output_file}")
        except Exception as e:
            print(f" Could not save timeline: {e}")
        
        plt.show()
    
    def visualize_network(self, G, leaders_df, output_file='coordination_network.png'):
        """Visualize coordination network"""
        print(f"\n🕸️ Creating network visualization...")
        
        if G is None or len(G.nodes()) == 0:
            print(" No network data to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node sizes based on betweenness centrality
        node_sizes = [leaders_df[leaders_df['page'] == node]['betweenness_centrality'].values[0] * 10000 
                      if len(leaders_df[leaders_df['page'] == node]) > 0 else 100 
                      for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#3498db', alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, edge_color='gray', ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        ax.set_title('Coordination Network Structure', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f" Network visualization saved: {output_file}")
        except Exception as e:
            print(f" Could not save network: {e}")
        
        plt.show()
    
    def save_results(self, coord_df, cai_df, leaders_df, output_file='coordination_analysis_results.xlsx'):
        """Save all results to Excel"""
        print(f"\n Saving results to: {output_file}")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Coordinated Pairs',
                    'Unique Pages Involved',
                    'Analysis Period (days)',
                    'Similarity Threshold',
                    'Time Window (minutes)',
                    'Max CAI Score',
                    'Analysis Date'
                ],
                'Value': [
                    len(coord_df) if len(coord_df) > 0 else 0,
                    len(set(coord_df['page_1'].tolist() + coord_df['page_2'].tolist())) if len(coord_df) > 0 else 0,
                    len(cai_df) if len(cai_df) > 0 else 0,
                    f"{self.similarity_threshold * 100}%",
                    self.time_window.total_seconds() / 60,
                    round(cai_df['CAI'].max(), 4) if len(cai_df) > 0 else 0,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Coordinated pairs
            if len(coord_df) > 0:
                coord_df.to_excel(writer, sheet_name='Coordinated_Pairs', index=False)
            
            # CAI timeline
            if len(cai_df) > 0:
                cai_df.to_excel(writer, sheet_name='CAI_Timeline', index=False)
            
            # Network leaders
            if leaders_df is not None and len(leaders_df) > 0:
                leaders_df.to_excel(writer, sheet_name='Network_Leaders', index=False)
        
        print(" Results saved successfully!")
    
    def run_full_analysis(self, excel_file, output_file='coordination_analysis_results.xlsx'):
        """Run complete coordination analysis pipeline"""
        print("\n" + "=" * 70)
        print(" COORDINATION ACTIVATION INDEX (CAI) ANALYSIS")
        print("   For Georgian Pro-Establishment Disinformation Network")
        print("=" * 70)
        
        try:
            # Load data
            df = self.load_and_prepare_data(excel_file)
            
            # Detect coordination
            coord_df = self.detect_coordinated_posts(df)
            
            # Calculate CAI
            cai_df = self.calculate_cai(coord_df, df)
            
            # Identify leaders
            leaders_df, G = self.identify_network_leaders(coord_df)
            
            # Visualizations
            if len(cai_df) > 0:
                self.visualize_cai_timeline(cai_df)
            
            if G is not None:
                self.visualize_network(G, leaders_df)
            
            # Save results
            self.save_results(coord_df, cai_df, leaders_df, output_file)
            
            print("\n" + "=" * 70)
            print(" ANALYSIS COMPLETE!")
            print("=" * 70)
            
            return {
                'coordinated_pairs': coord_df,
                'cai_timeline': cai_df,
                'network_leaders': leaders_df,
                'network_graph': G
            }
            
        except Exception as e:
            print(f"\n Analysis failed: {str(e)}")
            raise


def main():
    """Main execution function"""
    
     # REMOVED BECAUSE ITS NOT PUBLIC INFO!
    EXCEL_FILE = "C:\\Users\\HP\\OneDrive\\Desktop\\"
    OUTPUT_FILE = "C:\\Users\\HP\\OneDrive\\Desktop\\"
    
    # Initialize analyzer with research parameters
    analyzer = CoordinationAnalyzer(
        similarity_threshold=0.85,      # 85% content similarity
        time_window_minutes=30,         # 30-minute coordination window
        min_word_length=10              # Minimum 10 words per post
    )
    
    # Run complete analysis
    results = analyzer.run_full_analysis(
        excel_file=EXCEL_FILE,
        output_file=OUTPUT_FILE
    )
    
    print("\n All results saved to:", OUTPUT_FILE)
    print(" Visualizations displayed and saved as PNG files")
    print("\n Ready for ForSet fellowship submission!")


if __name__ == "__main__":
    main()