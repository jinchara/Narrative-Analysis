#OPTIMIZED SIMILAR DATA SHEET
# ADDED ID COLUMN + NARRATIVE CLASSIFICATION + INTENSITY ANALYSIS
# Propaganda Channel Content Similarity Analysis
### GRAPH API & ONLY ISFED WRITTEN DATA
# Optimized for finding channels that share the same propaganda content
# Extended: narrative tagging, daily intensity, channel count per narrative
#min_similarity=0.50

import pandas as pd
import numpy as np
import time
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# NARRATIVE KEYWORD DEFINITIONS
# 6 narratives matching the media monitoring research
# ============================================================
NARRATIVE_KEYWORDS = {
    # ── ნარატივი 1 ──────────────────────────────────────────────────
    # ანტიდასავლური / Deep State
    "ანტიდასავლური / Deep State": [
        # ინგლისური ტერმინები
        "deep state", "usaid", "ned ", "ngo",
        # სუვერენიტეტი
        "სუვერენიტეტ", "სუვერენულ",
        # ჩარევა და ზეწოლა
        "ჩარევ", "ჩაერ", "ერევ", "მენტორ",
        "გარე ზეწოლ", "დასავლეთის ზეწოლ", "გარე ჩარევ",
        "უცხოური ზეწოლ", "უცხოური გავლენ", "უცხოური ჩარევ",
        # ბრიუსელი და ევროპის კრიტიკა (2026 ფოკუსი)
        "ბრიუსელ", "ევროპის კრიზის", "ევროპის სისუსტ",
        "ევროპის პრობლემ", "მიგრაციულ კრიზის",
        "ბიუროკრატ", "ევროპული ბიუროკრატ",
        "ევროკავშირს ბუმერანგ", "ბუმერანგის პრინციპ",
        "შეურაცხმყოფელი და უსამართლო ქმედებ",
        # ელჩები (2026-ში ირონიული ხსენება)
        "ელჩ", "საელჩ",
        # ღირებულებები / შემოჭრა
        "ტრადიციული იდენტობ", "ღირებულებების შეცვლ",
        "ორმაგი სტანდარტ", "მორალური დაკნინ",
        "შავი ფული", "შავ ფულ",
        # ტრამპი / მოკავშირე (2025 ფოკუსი)
        "ტრამპ", "თანამოაზრ",
        # დიპ სთეიტ ქართულად
        "დიპ სთეიტ", "ღრმა სახელმწიფო",
        # სისტემური პროცესი / ზეწოლა
        "სისტემური პროცეს", "ვიზო რეჟიმ", "სანქცი",
        "უვიზო რეჟიმ", "ვიზის გაუქმ",
        # ჩვენს საქმეში
        "ჩვენს საქმეში", "შიდა საქმეებ",
        # BBC/გრანტები/გავლენა
        "უცხოურ გავლენასთან ბრძოლ", "გავუზიაროთ გამოცდილება",
        "შევიტანეთ სარჩელი", "ბი-ბი-სი", "bbc",
        "გრანტების შესახებ", "გრანტების კანონ",
    ],

    # ── ნარატივი 2 ──────────────────────────────────────────────────
    # ოპოზიციის დელეგიტიმაცია
    "ოპოზიციის დელეგიტიმაცია": [
        # ნაცმოძრაობა — ყველა ფორმა
        "ნაცმოძრაობ", "ნაციონალური მოძრაობ",
        "ნაც ", "ნაცებ", "ნაცი ", "ნაცექსპერტ", "ნაცოპოზიც",
        # დეჰუმანიზაცია
        "ორკ", "ნაცორკ", "ვირთხ", "მღრღნელ",
        # კრიმინალიზაცია
        "მოღალატ", "მოღალ", "ღალატ",
        "კრიმინალ", "დამნაშავ", "კანონდამრღვ",
        "ქვეყნის მტერ", "სახელმწიფოს მტერ",
        # უცხოური კავშირი
        "უცხოური აგენტ", "დასავლეთის აგენტ",
        "მფარველ", "მათი მფარველ",
        # ძალადობა და მუქარა
        "რადიკალ", "ექსტრემ", "ძალადობ",
        "პროვოკაც", "ბრბო",
        # სცენარი
        "რევოლუცი", "გადატრიალ", "პუტჩ",
        "ოპოზიცია გეგმ", "ოპოზიცია აწყობ",
        "ოპოზიციური სცენარ",
        # კონკრეტული ოპოზიციური ფიგურები
        "ელისაშვილ", "გვარამია", "ზურაბიშვილ",
        "სააკაშვილ", "სანიკიძ",
        # საია / არასამთავრობოები
        "საია-მ", "საია ", "სპექტაკლი შემოგვთავაზ",
        # "მაიდანი / ვენესუელა" ფრეიმინგი
        "მაიდანს", "ვენესუელას ნატრ", "ნეპალს ნატრ",
        "ქვეყნის ნგრევ", "ქვეყნის დანგრევ",
        # ოპოზ. მედია
        "მთავარი არხ", "TV პირველ", "ტელეკომპანია პირველ",
    ],

    # ნარატივი 3
    # ევროსკეპტიციზმი / მორალური დაკნინება
    "ევროსკეპტიციზმი": [
        # LGBTQ
        "lgbtq", "lgbt", "ლგბტ", "ლგბტქ",
        "სქესობრივი", "გენდერ", "ტრანსგენდ",
        "სექსუალური უმცირესობ", "ჰომოსექსუალ",
        "გეი პარად", "სიამაყის მარშ",
        # ეროვნული იდენტობა
        "ეროვნული იდენტობ", "ეროვნული ღირებულ",
        "ქართული ღირებულ", "ქრისტიანულ ღირებულ",
        "ტრადიციულ ღირებულ", "ტრადიციულ ოჯახ",
        # ევროკავშირის კრიტიკა
        "ევროკავშირი ვერ", "ევროკავშირს არ",
        "ევროკავშირის კრიზის", "ევრო კრიზის",
        "ანტისახალხო", "ევროპის ანტი",
        "ევროპული ჰიპოკრიზი",
        "სამარცხვინო", "შეურაცხმყოფ",
        # ეკომისარი / "ხვრელები" (ნამდვილი ტექსტებიდან)
        "ხვრელებს ეძებ", "ეძებენ გზებსა და ხვრელ",
        "ევროკომისარ",
        # დემოგრაფია / რელიგია
        "დემოგრაფიულ კრიზის", "მოსახლეობის კლება",
        "მართლმადიდებლ", "ეკლესი", "პატრიარქ",
        "ქრისტიანობ",
    ],

    # ნარატივი 4
    # მთავრობის წარმატება / ეკონომიკური აღმავლობა
    "მთავრობის წარმატება": [
        # ეკონომიკა
        "ეკონომიკური ზრდ", "ეკონომიკა იზრდ",
        "ორნიშნა", "მშპ", "ეკონომიკური წარმატ",
        # ინვესტიციები
        "ინვესტიცი", "eagle hills", "ინვესტორ",
        "საინვესტიციო პროექტ", "მილიარდ",
        # ინფრასტრუქტურა
        "ინფრასტრუქტურ", "გზის მშენებლ", "ახალი გზ",
        "ახალი ხიდ", "ახალი სკოლ", "ახალი საავადმყოფ",
        # ზოგადი წარმატება
        "წარმატებ", "განვითარებ", "პროგრეს",
        "გამარჯვება ეკონომ", "რეკორდ",
        "ქვეყანა ვითარდებ",
        # სოციალური
        "სოციალური პროგრამ", "პენსი", "ხელფასი გაიზარდ",
        "დასაქმებ", "სამუშაო ადგილ",
        # ფასნამატი / ფასების კონტროლი (2026 კობახიძის ინიციატივა — ნამდვილი ტექსტებიდან)
        "ფასნამატ", "ფასნამატის წარმოქმნ", "ფასებზე კონტრ",
        "ფასები შემცირდ", "სადაც ფასები",
        "პრემიერი: მაქსიმუმ", "აპრილის ბოლომდე",
        # ინტერნეტი / ტექნოლოგია
        "მობილური ინტერნეტ", "სისწრაფით მსოფლიოში",
        "ინტერნეტის სისწრაფ",
        # სპორტი / ოლიმპიადა
        "ოლიმპიელ", "სპორტსმენ", "ჩემპიონ",
        "გამარჯვებისთვის ჩვენს ქართველ",
        # ფულადი გზავნილები
        "ფულადი გზავნილ", "გზავნილების ნაკად",
        # სამართალდამცავები
        "წარმატებ დანაშაულთ", "ყველაზე წარმატებ",
        "ეფექტიანობ", "სამართალდამცავ",
    ],

    # ნარატივი 5
    # ისტორიული რევიზიონიზმი (2004-2012)
    "ისტორიული რევიზიონიზმი": [
        # კომისია
        "საგამოძიებო კომისი", "გამოძიების კომისი",
        # წლები
        "2004 წელ", "2005 წელ", "2006 წელ",
        "2007 წელ", "2008 წელ", "2009 წელ",
        "2010 წელ", "2011 წელ", "2012 წელ",
        "2004-2012",
        # ძალადობა
        "წამებ", "წამება", "პატიმრობ",
        "სასჯელაღსრულებ", "ციხეში", "ციხის",
        "სიკვდილი პატიმ", "ყოფილ ციხ",
        # რეჟიმი
        "ნაციონალების დანაშაულ", "ნაციონალების რეჟიმ",
        "ნაციონალური მოძრაობის დანაშ",
        "ნაციონალური მოძრაობის მმართ",
        "ყოფილი ხელისუფლებ", "მმართველობის წლებ",
        "ის ხელისუფლებ",
        # მედია / სასამართლო კონტროლი
        "მედიის კონტროლ", "სასამართლოს კონტროლ",
    ],

    # ნარატივი 6
    # არჩევნების ლეგიტიმურობა (2025) + სახელმწიფო რეფორმები (2026)
    "არჩევნების ლეგიტიმურობა / სახელმწიფო რეფორმები": [
        # არჩევნები (2025 ფოკუსი)
        "არჩევნებ", "საარჩევნო", "ხმის დათვლ",
        "ხმის მიცემ", "ბიულეტენ",
        "დამკვირვებელ", "საერთაშორისო დამკვირვ",
        "ლეგიტიმ", "ლეგიტიმური",
        "4 ოქტომბ", "26 ოქტომბ",
        # მანდატი (2026 ფოკუსი)
        "მანდატ", "ხალხის მანდატ", "ხალხმა მხარი",
        "გავიმარჯვ", "ხალხის ნება",
        "ამომრჩევლის დაკვეთ",  # ნამდვილი ციტატა: "ჩვენ გვაქვს ჩვენი ამომრჩევლის დაკვეთა"
        # სახელმწიფო რეფორმები (2026 ახალი ნარატივი)
        "განათლების რეფორმ", "სასკოლო რეფორმ",
        "ფასების რეგულირ", "ფასების კონტრ",
        "სახელმწიფო ცვლილებ", "ფუნდამენტური ცვლილებ",
        "ახალი კანონ", "კანონპროექტ",
        "სახელმწიფო რეფორმ", "მმართველობ",
        "სისტემური ცვლილებ", "ინსტიტუციური ცვლილ",
        # უნივერსიტეტების გაერთიანება (2026 კონკრეტული პოლიტიკა)
        "თსუ-ს და სტუ-ს გაერთიანებ", "უნივერსიტეტის გაერთიანებ",
        "კვოტების საკითხ",
        # მანქანების შემოყვანის კანონი (2026 კონკრეტული)
        "ექვს წელზე მეტი წლოვანების მანქან",
        "მანქანების შემოყვანის აკრძალ",
    ],
}

def classify_narrative(text):
    """
    Classify text into one or more narratives based on keyword matching.
    Returns a list of matched narrative names.
    """
    if not text or not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    matched = []
    for narrative, keywords in NARRATIVE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                matched.append(narrative)
                break  # one match per narrative is enough
    
    return matched if matched else ["სხვა / კლასიფიკაცია არ მოხდა"]


class PropagandaAnalyzer:
    def __init__(self, min_similarity=0.50, min_word_length=10, max_length_ratio=3.0,
                 account_filter=None, start_date=None, end_date=None):
        self.min_similarity = min_similarity
        self.min_word_length = min_word_length
        self.max_length_ratio = max_length_ratio
        self.account_filter = account_filter
        self.start_date = start_date
        self.end_date = end_date
        self.vectorizer = None

    def read_data(self, excel_file):
        print(f"📖 Reading data from: {excel_file}")
        try:
            df = pd.read_excel(excel_file)
            print(f"✅ Successfully read Excel file")
        except Exception as e:
            raise Exception(f" Could not read the Excel file: {str(e)}")

        required_columns = ['id', 'post_owner.name', 'text', 'creation_time']
        if self.account_filter:
            required_columns.append('Account')

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise Exception(f" Missing columns: {missing_columns}")

        print(f" Loaded {len(df)} total records")
        return df

    def clean_and_prepare_data(self, df):
        print(" Cleaning and preparing data...")
        df = df.copy()

        if self.account_filter:
            initial_count = len(df)
            df = df[df['Account'].astype(str).str.contains(
                self.account_filter, case=False, na=False)].copy()
            print(f" Filtered by 'Account': {len(df)} records remaining (removed {initial_count - len(df)})")

        df['clean_content'] = df['text'].fillna('').astype(str).str.strip()
        df['clean_content'] = df['clean_content'].str.replace(r'<[^>]+>', '', regex=True)
        df['clean_content'] = df['clean_content'].str.replace(r'\s+', ' ', regex=True)
        df['clean_content'] = df['clean_content'].str.lower()
        df['word_count'] = df['clean_content'].str.split().str.len()

        initial_count = len(df)
        df = df[df['word_count'] >= self.min_word_length].copy()
        df = df[df['clean_content'].str.len() > 20].copy()
        df = df.drop_duplicates(subset=['post_owner.name', 'clean_content']).copy()
        df = df.reset_index(drop=True)

        # Parse dates
        df['creation_time'] = pd.to_datetime(df['creation_time'], errors='coerce')
        df['date_only'] = df['creation_time'].dt.date

        # === NARRATIVE CLASSIFICATION ===
        print(" Classifying narratives...")
        df['narratives'] = df['text'].apply(classify_narrative)
        df['narrative_primary'] = df['narratives'].apply(
            lambda x: x[0] if x else "სხვა / კლასიფიკაცია არ მოხდა")
        df['narrative_count'] = df['narratives'].apply(len)

        print(f" After cleaning: {len(df)} records (removed {initial_count - len(df)})")
        print(f" Found {df['post_owner.name'].nunique()} unique channels")

        # Narrative distribution summary
        exploded = df.explode('narratives')
        narrative_counts = exploded['narratives'].value_counts()
        print("\n Narrative distribution:")
        for narr, count in narrative_counts.items():
            print(f"   {narr}: {count} posts")

        return df

    def create_content_hash(self, text):
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return hashlib.md5(clean_text.encode()).hexdigest()

    def find_exact_matches(self, df):
        print(" Finding exact and near-exact matches...")
        df['content_hash'] = df['clean_content'].apply(self.create_content_hash)
        hash_groups = df.groupby('content_hash')

        exact_matches = []
        for hash_val, group in hash_groups:
            if len(group) > 1:
                channels = group['post_owner.name'].unique()
                if len(channels) > 1:
                    narratives_in_group = []
                    for narr_list in group['narratives']:
                        narratives_in_group.extend(narr_list)
                    primary_narrative = (
                        max(set(narratives_in_group), key=narratives_in_group.count)
                        if narratives_in_group else "სხვა"
                    )
                    exact_matches.append({
                        'content_hash': hash_val,
                        'content_sample': group.iloc[0]['text'],
                        'channels': list(channels),
                        'channel_count': len(channels),
                        'total_posts': len(group),
                        'similarity_type': 'Exact Match',
                        'primary_narrative': primary_narrative,
                        'all_narratives': ' | '.join(set(narratives_in_group)),
                        'dates': list(group['creation_time'].unique()),
                        'date_only': list(group['date_only'].unique()),
                        'ids': list(group['id'].astype(str).unique())
                    })

        print(f" Found {len(exact_matches)} exact match groups")
        return exact_matches

    def find_similar_content_clusters(self, df):
        print(" Analyzing content similarity with TF-IDF...")
        df_for_similarity = df.copy()
        if len(df_for_similarity) < 2:
            return []

        self.vectorizer = TfidfVectorizer(
            max_features=5000, min_df=2, max_df=0.8,
            ngram_range=(1, 3), sublinear_tf=True)

        try:
            tfidf_matrix = self.vectorizer.fit_transform(df_for_similarity['clean_content'])
            print(f" Created TF-IDF matrix: {tfidf_matrix.shape}")
        except Exception as e:
            print(f" Error creating TF-IDF matrix: {e}")
            return []

        print(" Computing content similarities...")
        similarity_matrix = cosine_similarity(tfidf_matrix)

        similar_groups = []
        processed_indices = set()

        for i in range(len(df_for_similarity)):
            if i in processed_indices:
                continue

            similar_indices = [
                j for j in range(i, len(df_for_similarity))
                if j not in processed_indices and similarity_matrix[i, j] >= self.min_similarity
            ]

            if len(similar_indices) > 1:
                cluster_data = df_for_similarity.iloc[similar_indices]
                channels = cluster_data['post_owner.name'].unique()

                if len(channels) > 1:
                    group_similarities = [
                        similarity_matrix[similar_indices[idx1], similar_indices[idx2]]
                        for idx1 in range(len(similar_indices))
                        for idx2 in range(idx1 + 1, len(similar_indices))
                    ]
                    avg_similarity = np.mean(group_similarities) if group_similarities else 0

                    word_counts = cluster_data['word_count'].values
                    max_ratio = (np.max(word_counts) / np.min(word_counts)
                                 if np.min(word_counts) > 0 else float('inf'))

                    if avg_similarity >= self.min_similarity and max_ratio <= self.max_length_ratio:
                        if avg_similarity < 0.95 or max_ratio > 1.1:
                            narratives_in_group = []
                            for narr_list in cluster_data['narratives']:
                                narratives_in_group.extend(narr_list)
                            primary_narrative = (
                                max(set(narratives_in_group), key=narratives_in_group.count)
                                if narratives_in_group else "სხვა"
                            )

                            similar_groups.append({
                                'group_id': len(similar_groups),
                                'content_sample': cluster_data.iloc[0]['text'],
                                'channels': list(channels),
                                'channel_count': len(channels),
                                'total_posts': len(cluster_data),
                                'avg_similarity': round(avg_similarity, 3),
                                'max_length_ratio': round(max_ratio, 2),
                                'similarity_type': 'High Similarity',
                                'primary_narrative': primary_narrative,
                                'all_narratives': ' | '.join(set(narratives_in_group)),
                                'dates': list(cluster_data['creation_time'].unique()),
                                'date_only': list(cluster_data['date_only'].unique()),
                                'word_counts': list(cluster_data['word_count'].values),
                                'ids': list(cluster_data['id'].astype(str).unique())
                            })
                        processed_indices.update(similar_indices)

        print(f" Found {len(similar_groups)} high similarity groups")
        return similar_groups

    def analyze_channel_relationships(self, exact_matches, similar_groups):
        print(" Analyzing channel relationships...")
        channel_connections = defaultdict(set)
        channel_stats = defaultdict(lambda: {'exact_matches': 0, 'similar_matches': 0, 'total_shared': 0})

        for match in exact_matches + similar_groups:
            channels = match['channels']
            match_type = 'exact_matches' if match['similarity_type'] == 'Exact Match' else 'similar_matches'
            for i, ch1 in enumerate(channels):
                channel_stats[ch1][match_type] += 1
                channel_stats[ch1]['total_shared'] += 1
                for j, ch2 in enumerate(channels):
                    if i != j:
                        channel_connections[ch1].add(ch2)

        channel_summary = [{
            'Channel': str(ch),
            'Exact_Matches': int(stats['exact_matches']),
            'Similar_Matches': int(stats['similar_matches']),
            'Total_Shared_Content': int(stats['total_shared']),
            'Connected_Channels': int(len(channel_connections[ch])),
            'Connected_Channel_List': ', '.join(sorted(str(c) for c in channel_connections[ch]))
        } for ch, stats in channel_stats.items()]

        channel_summary.sort(key=lambda x: int(x['Total_Shared_Content']), reverse=True)
        return channel_summary

    # ============================================================
    # NEW: NARRATIVE INTENSITY ANALYSIS
    # For each narrative + date: how many unique channels posted it
    # This is the "average channels per narrative" data for charts
    # ============================================================
    def build_narrative_intensity(self, df):
        """
        For each (date, narrative) pair: count unique channels that posted content
        classified under that narrative on that date.
        Returns a pivot table: rows=date, columns=narrative, values=unique_channel_count
        """
        print("📅 Building narrative intensity over time...")
        
        # Explode so each row = one post + one narrative
        exploded = df.explode('narratives').copy()
        exploded = exploded[exploded['narratives'] != "სხვა / კლასიფიკაცია არ მოხდა"]
        exploded = exploded.dropna(subset=['date_only', 'narratives'])

        # Per day + narrative: unique channels
        intensity = (
            exploded.groupby(['date_only', 'narratives'])['post_owner.name']
            .nunique()
            .reset_index()
        )
        intensity.columns = ['date', 'narrative', 'unique_channels']

        # Pivot: rows=date, columns=narrative
        pivot = intensity.pivot_table(
            index='date', columns='narrative',
            values='unique_channels', aggfunc='sum', fill_value=0
        ).reset_index()

        print(f" Intensity table: {len(pivot)} days × {len(pivot.columns)-1} narratives")
        return intensity, pivot

    def build_narrative_summary(self, df, exact_matches, similar_groups):
        """
        Per narrative:
        - total posts classified
        - unique channels that posted
        - average unique channels per day (= intensity metric from the article)
        - total coordinated groups (exact + similar) tagged with this narrative
        """
        print(" Building narrative summary stats...")

        # Post-level stats
        exploded = df.explode('narratives').copy()
        exploded = exploded[exploded['narratives'] != "სხვა / კლასიფიკაცია არ მოხდა"]

        post_stats = (
            exploded.groupby('narratives')
            .agg(
                total_posts=('id', 'count'),
                unique_channels=('post_owner.name', 'nunique'),
            )
            .reset_index()
        )

        # Daily average unique channels per narrative
        daily = (
            exploded.groupby(['date_only', 'narratives'])['post_owner.name']
            .nunique()
            .reset_index()
        )
        daily_avg = (
            daily.groupby('narratives')['post_owner.name']
            .mean()
            .reset_index()
            .rename(columns={'post_owner.name': 'avg_daily_channels'})
        )
        daily_avg['avg_daily_channels'] = daily_avg['avg_daily_channels'].round(2)

        # Coordinated group count per narrative
        all_matches = exact_matches + similar_groups
        narr_group_count = defaultdict(int)
        for m in all_matches:
            pn = m.get('primary_narrative', 'სხვა')
            narr_group_count[pn] += 1

        group_df = pd.DataFrame([
            {'narratives': k, 'coordinated_groups': v}
            for k, v in narr_group_count.items()
        ])

        # Merge all
        summary = post_stats.merge(daily_avg, on='narratives', how='left')
        if not group_df.empty:
            summary = summary.merge(group_df, on='narratives', how='left')
        else:
            summary['coordinated_groups'] = 0

        summary['coordinated_groups'] = summary['coordinated_groups'].fillna(0).astype(int)
        summary = summary.sort_values('avg_daily_channels', ascending=False)
        summary.rename(columns={'narratives': 'narrative'}, inplace=True)

        return summary

    def build_channel_narrative_matrix(self, df):
        """
        Matrix: rows=channel, columns=narrative, values=post_count
        Useful for heatmap visualization.
        """
        print(" Building channel × narrative matrix...")
        exploded = df.explode('narratives').copy()
        exploded = exploded[exploded['narratives'] != "სხვა / კლასიფიკაცია არ მოხდა"]

        matrix = exploded.pivot_table(
            index='post_owner.name', columns='narratives',
            values='id', aggfunc='count', fill_value=0
        ).reset_index()
        matrix.rename(columns={'post_owner.name': 'Channel'}, inplace=True)
        return matrix

    def build_coordinated_groups_by_date(self, exact_matches, similar_groups):
        """
        Daily count of coordinated groups (exact + similar), with narrative tag.
        This gives the timeline chart data.
        """
        records = []
        for m in exact_matches + similar_groups:
            for d in m.get('date_only', []):
                records.append({
                    'date': d,
                    'narrative': m.get('primary_narrative', 'სხვა'),
                    'similarity_type': m['similarity_type'],
                    'channel_count': m['channel_count']
                })

        if not records:
            return pd.DataFrame()

        df_coord = pd.DataFrame(records)
        df_coord['date'] = pd.to_datetime(df_coord['date'])

        daily_coord = (
            df_coord.groupby(['date', 'narrative', 'similarity_type'])
            .agg(group_count=('channel_count', 'count'),
                 avg_channels=('channel_count', 'mean'))
            .reset_index()
        )
        daily_coord['avg_channels'] = daily_coord['avg_channels'].round(2)
        return daily_coord

    def save_results(self, exact_matches, similar_groups, channel_summary,
                     narrative_intensity_long, narrative_intensity_pivot,
                     narrative_summary, channel_narrative_matrix,
                     daily_coord, output_file):
        print(f"💾 Saving results to: {output_file}")

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

            # Sheet 1: Summary
            summary_data = {
                'Metric': [
                    'Total Exact Match Groups', 'Total Similar Content Groups',
                    'Total Channels Analyzed', 'Channels with Shared Content',
                    'Analysis Threshold', 'Analysis Date', 'Account Filter Used'
                ],
                'Value': [
                    len(exact_matches), len(similar_groups),
                    len(channel_summary),
                    len([c for c in channel_summary if c['Total_Shared_Content'] > 0]),
                    f"{self.min_similarity*100}%",
                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    self.account_filter if self.account_filter else 'None'
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            # Sheet 2: Channel Analysis
            if channel_summary:
                pd.DataFrame(channel_summary).to_excel(
                    writer, sheet_name='Channel_Analysis', index=False)

            # Sheet 3: Exact Matches
            if exact_matches:
                exact_df = pd.DataFrame(exact_matches)
                for col in ['channels', 'dates', 'ids', 'date_only']:
                    if col in exact_df.columns:
                        exact_df[col] = exact_df[col].apply(
                            lambda x: ' | '.join(map(str, x)))
                exact_df.to_excel(writer, sheet_name='Exact_Matches', index=False)

            # Sheet 4: Similar Content Groups
            if similar_groups:
                similar_df = pd.DataFrame(similar_groups)
                for col in ['channels', 'dates', 'word_counts', 'ids', 'date_only']:
                    if col in similar_df.columns:
                        similar_df[col] = similar_df[col].apply(
                            lambda x: ' | '.join(map(str, x)))
                similar_df.to_excel(writer, sheet_name='Similar_Groups', index=False)

            # ---- NEW SHEETS FOR VISUALIZATION ----

            # Sheet 5: Narrative Summary (bar chart ready)
            # avg_daily_channels = the "საშუალო არხების რაოდენობა" metric from the article
            if not narrative_summary.empty:
                narrative_summary.to_excel(
                    writer, sheet_name='Narrative_Summary', index=False)

            # Sheet 6: Daily Intensity (line chart ready)
            # rows: date | narrative | unique_channels
            if not narrative_intensity_long.empty:
                narrative_intensity_long.to_excel(
                    writer, sheet_name='Daily_Intensity_Long', index=False)

            # Sheet 7: Daily Intensity Pivot (date × narrative)
            # Each column = one narrative, each row = one day
            if not narrative_intensity_pivot.empty:
                narrative_intensity_pivot.to_excel(
                    writer, sheet_name='Daily_Intensity_Pivot', index=False)

            # Sheet 8: Channel × Narrative Matrix (heatmap ready)
            if not channel_narrative_matrix.empty:
                channel_narrative_matrix.to_excel(
                    writer, sheet_name='Channel_Narrative_Matrix', index=False)

            # Sheet 9: Coordinated Groups by Date + Narrative (timeline chart)
            if not daily_coord.empty:
                daily_coord.to_excel(
                    writer, sheet_name='Coordinated_Groups_Daily', index=False)

        print(" Results saved successfully!")
        print("\n NEW SHEETS FOR VISUALIZATION:")
        print("   • Narrative_Summary         → bar chart: avg channels per narrative")
        print("   • Daily_Intensity_Long      → line chart: intensity over time per narrative")
        print("   • Daily_Intensity_Pivot     → pivot table: date × narrative (for Flourish)")
        print("   • Channel_Narrative_Matrix  → heatmap: which channels push which narratives")
        print("   • Coordinated_Groups_Daily  → timeline: coordinated groups count per day")

    def run_analysis(self, excel_file, output_file):
        start_time = time.time()
        print(" Starting Propaganda Channel Analysis (Extended)...")
        print("="*60)

        try:
            df = self.read_data(excel_file)
            df = self.clean_and_prepare_data(df)

            if len(df) == 0:
                print(" No data to analyze after cleaning and filtering")
                return None

            exact_matches = self.find_exact_matches(df)
            similar_groups = self.find_similar_content_clusters(df)
            channel_summary = self.analyze_channel_relationships(exact_matches, similar_groups)

            # NEW: narrative-level analysis
            narrative_intensity_long, narrative_intensity_pivot = self.build_narrative_intensity(df)
            narrative_summary = self.build_narrative_summary(df, exact_matches, similar_groups)
            channel_narrative_matrix = self.build_channel_narrative_matrix(df)
            daily_coord = self.build_coordinated_groups_by_date(exact_matches, similar_groups)

            self.save_results(
                exact_matches, similar_groups, channel_summary,
                narrative_intensity_long, narrative_intensity_pivot,
                narrative_summary, channel_narrative_matrix,
                daily_coord, output_file
            )

            print("\n" + "="*60)
            print(" ANALYSIS RESULTS SUMMARY")
            print("="*60)
            print(f" Total channels analyzed:        {len(channel_summary)}")
            print(f" Exact match groups found:        {len(exact_matches)}")
            print(f" Similar content groups found:    {len(similar_groups)}")

            if not narrative_summary.empty:
                print("\n Narrative ranking by avg daily channels:")
                for _, row in narrative_summary.iterrows():
                    print(f"   {row['narrative'][:50]:<52} avg {row['avg_daily_channels']} channels/day")

            if channel_summary:
                print("\n Top 5 channels with most shared content:")
                for i, ch in enumerate(channel_summary[:5], 1):
                    print(f"  {i}. {ch['Channel']}: {ch['Total_Shared_Content']} shared items")

            elapsed = time.time() - start_time
            print(f"\n Completed in {elapsed:.2f}s ({elapsed/60:.1f} min)")

            return {
                'exact_matches': exact_matches,
                'similar_groups': similar_groups,
                'channel_summary': channel_summary,
                'narrative_summary': narrative_summary,
                'processing_time': elapsed
            }

        except Exception as e:
            print(f" Error during analysis: {str(e)}")
            raise


def main():
    # REMOVED BECAUSE ITS NOT PUBLIC INFO!
    EXCEL_FILE  = "C:\\Users\\HP\\OneDrive\\Desktop\\"
    OUTPUT_FILE = "C:\\Users\\HP\\OneDrive\\Desktop\\"


    analyzer = PropagandaAnalyzer(
        min_similarity=0.50,
        min_word_length=10,
        max_length_ratio=3.0,
        account_filter="Info"
    )

    results = analyzer.run_analysis(EXCEL_FILE, OUTPUT_FILE)

    if results:
        print("\n Analysis completed successfully!")
        print(f" Results saved to: {OUTPUT_FILE}")
    else:
        print(" Analysis failed or no results found")


if __name__ == "__main__":
    main()