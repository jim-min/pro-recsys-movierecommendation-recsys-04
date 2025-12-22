import os
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional

@st.cache_data
def load_ratings(data_dir: str) -> Optional[pd.DataFrame]:
    """Loads train_ratings.csv"""
    file_path = os.path.join(data_dir, 'train_ratings.csv')
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    
    try:
        # Check delimiter - assuming csv implies comma, but recsys data sometimes uses tabs or colons
        # Based on inspection earlier using `head`: "11,4643,1230782529" -> Comma separated
        df = pd.read_csv(file_path)
        
        # Ensure time is proper type if needed, but int/float is fine for sorting
        # Explicitly converting to datetime might be useful for plotting
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            
        return df
    except Exception as e:
        st.error(f"Error loading ratings: {e}")
        return None

@st.cache_data
def load_item_info(data_dir: str) -> pd.DataFrame:
    """Loads and merges item information from tsv files."""
    
    # Files to look for
    info_files = {
        'titles': 'titles.tsv',
        'genres': 'genres.tsv', 
        'directors': 'directors.tsv',
        'writers': 'writers.tsv',
        'years': 'years.tsv'
    }
    
    merged_df = None
    
    for key, filename in info_files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue
            
        try:
            # Inspection showed titles.tsv: "item\ttitle" -> Tab separated
            df = pd.read_csv(path, sep='\t')
            
            # Ensure 'item' column is index or key for merging
            if 'item' not in df.columns:
                continue
                
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='item', how='outer')
                
        except Exception as e:
            st.warning(f"Failed to load {filename}: {e}")
            
    if merged_df is None:
        return pd.DataFrame(columns=['item'])
        
    return merged_df
