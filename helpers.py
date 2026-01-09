import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import matplotlib as mpl
import json

from collections import Counter

import warnings
warnings.filterwarnings('ignore')

def count_list_of_list(list_of_lists):
    """
    Input: List of list of keywords
    Output: {keyword: count} dict
    """
    counter = Counter()

    # Allowed characters: letters, digits, spaces, and selected punctuation
    allowed_pattern = r"[a-z0-9\s\-/.,&'():]+"

    for sublist in list_of_lists:
        cleaned_items = set()
        for word in sublist:
            if not isinstance(word, str):
                continue
            w = word.strip().lower()

            if re.fullmatch(allowed_pattern, w):
                cleaned_items.add(w)

        counter.update(cleaned_items)

    return dict(counter)

def count_list(words):

    cleaned = [
        w.lower()
        for w in words
        if isinstance(w, str) and len(w.strip()) >= 1
    ]
    
    # Count the terms
    return dict(Counter(cleaned))

def count_unique_keywords(keywords):
    if not keywords:
        return {}

    # Case 1: List[str]
    if isinstance(keywords[0], str):
        return count_list(keywords)

    # Case 2: List[List[str]]
    if isinstance(keywords[0], (list, np.ndarray)):
        return count_list_of_list(keywords)

    # Otherwise raise error
    raise TypeError(
        f"Invalid keywords type: expected List[str] or List[List[str]], "
        f"got element of type {type(keywords[0])}"
    )


def counter_to_df(word_counts, store_name):
    df = pd.DataFrame(word_counts.items(), columns=['keyword', 'count'])
    df = df.sort_values(by='count', ascending=False, ignore_index=True)
    df['store_name'] = store_name
    return df

def item_list_to_df(store_keywords, store_name):
    word_count_dict = count_unique_keywords(store_keywords)
    res = counter_to_df(word_count_dict, store_name)
    return res
    
def join_keywords_in_list(list_of_lists, special_color="green", max_words=2):
    result = []
    for colors in list_of_lists:
        # Always include special_color if present
        has_special = special_color in colors

        # Take first max_words items, but ensure special_color is kept
        subset = colors[:max_words]
        if has_special and special_color not in subset:
            # Replace last element with special_color (preserve max length)
            if len(subset) == max_words:
                subset[-1] = special_color
            else:
                subset.append(special_color)

        # ✅ Only ensure special_color comes first, preserve original order otherwise
        subset = list(subset)
        subset.sort(key=lambda c: (c != special_color, c))

        # Join and wrap in list
        result.append([" ".join(subset)])
    return result


""" Add param to allow for combining of keywords """
def keyword_count_store_key_df(df, key, store_name, join_keywords = False):
    # Get [['cypress', 'fir'], ['magnolia']] 
    store_df = df[df.store_name == store_name]
    store_keywords =  store_df[key].values.tolist()

    # if flatten: store_keywords = ...
    if join_keywords:
        store_keywords = join_keywords_in_list(store_keywords)

    res = item_list_to_df(store_keywords, store_name)
    return res

def minmax_scale(data):
    """
    Scales a list or numpy array of numbers to the range [0, 100].
    Handles edge cases like constant values safely.
    """
    arr = np.array(data, dtype=float)
    min_val, max_val = np.min(arr), np.max(arr)

    # Avoid division by zero if all values are equal
    if min_val == max_val:
        return np.zeros_like(arr)

    scaled = (arr - min_val) / (max_val - min_val) * 100
    return scaled

def pivot_keyword_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        
    if not dfs:
        return pd.DataFrame()

    # Combine all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)

    # Pivot table: keywords as rows, store_names as columns
    pivot_df = merged_df.pivot_table(
        index='keyword',
        columns='store_name',
        values='count',
        aggfunc='sum',  # if duplicates exist, sum counts
        fill_value=0
    )
    
    return pivot_df

def get_pct_stores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'pct_stores' column showing the fraction of stores
    (columns) where the keyword has a count > 0. 
    Then sorts by pct_stores descending, then by row sum of counts descending.
    """
    pivot_df = df.copy()
    store_count = pivot_df.shape[1]
    
    # Compute pct_stores
    pivot_df['pct_stores'] = (pivot_df.gt(0).sum(axis=1) / store_count).round(2)
    
    # Compute row sum across numeric columns excluding pct_stores
    numeric_cols = pivot_df.select_dtypes(include='number').columns.tolist()
    sum_cols = [col for col in numeric_cols if col != 'pct_stores']
    
    item_totals = pivot_df[sum_cols].sum(axis=1).values
    pivot_df['score'] = minmax_scale(item_totals)/100
    
    # Sort first by pct_stores, then by row sum
    pivot_df = pivot_df.sort_values(by=['score', 'pct_stores'], ascending=False)
    pivot_df = pivot_df[[c for c in pivot_df.columns if c != 'pct_stores'] + ['pct_stores']]
    
    return pivot_df

def divide_df_by_store_sizes(df, store_sizes):
    print(df.columns)
    if len(df.columns) != len(store_sizes):
        raise ValueError("❌ Length of store_sizes must match number of DataFrame columns.")
    
    # Convert to Series for broadcasting
    divisors = pd.Series(store_sizes, index=df.columns)
    return df.divide(divisors, axis='columns')

def normalize_store_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Exclude non-numeric columns like 'pct_stores' if present
    numeric_df = df.select_dtypes(include='number')

    # Compute column-wise percentages
    pct_df = numeric_df.div(numeric_df.sum(axis=0), axis=1).round(2)

    # Retain non-numeric columns if any
    non_numeric = df.drop(columns=numeric_df.columns, errors='ignore')
    result = pd.concat([pct_df, non_numeric], axis=1)

    return result

def normalize_pivot_df(df, store_sizes, how = "store_size"):
    # print(df)
    if how == "store_size":
        return divide_df_by_store_sizes(df, store_sizes)
    else:
        return normalize_store_columns(df)

def generate_pivots(dfs, store_sizes, store_names = None, how = "store_size"):
    pivot_df = pivot_keyword_dfs(dfs)
    pivot_df = pivot_df[store_names]

    # EITHER BY STORE SIZE OF NORMALIZE
    pivot_df_pcts = normalize_pivot_df(pivot_df, store_sizes, how = how)

    pivot_df = pivot_df[store_names]
    pivot_df_pcts = pivot_df_pcts[store_names]
    
    return pivot_df, pivot_df_pcts

def aggregate_count_dfs(dfs, store_sizes, store_names = None, how = "store_size"):
    pivot_df, pivot_df_pcts = generate_pivots(dfs, store_sizes, store_names, how)
    # print(pivot_df)
    # MAKE SURE STORE COLUMNS MATCH

    count_df = get_pct_stores(pivot_df)
    pct_df = get_pct_stores(pivot_df_pcts)

    count_df = count_df[count_df.pct_stores > 0]
    pct_df = pct_df[pct_df.pct_stores > 0]

    return count_df, pct_df

def colorize_df(df: pd.DataFrame, base_cmap: str = "Greens", min_frac: float = 0.0, max_frac: float = 0.6):
    # Get original colormap
    cmap_orig = mpl.cm.get_cmap(base_cmap)
    
    # Slice colormap to soften extremes
    cmap_vals = cmap_orig(np.linspace(min_frac, max_frac, 256))
    # Remove alpha channel for pandas
    cmap_vals = cmap_vals[:, :-1]
    soft_cmap = mpl.colors.ListedColormap(cmap_vals)

    return (
        df.style
        .background_gradient(cmap=soft_cmap, vmin=0, vmax=100, axis=None)
        .format("{:.0f}%")
    )

def generate_color_map(dfs, store_sizes, store_names, how = "store_size"):
    count_pivot, pct_pivot = aggregate_count_dfs(dfs, store_sizes, store_names, how)
    return count_pivot, pct_pivot, colorize_df(pct_pivot*100)

@st.cache_data
def df_to_colortable_data(df, key, how='store_size', join_keywords=False):
    store_names = df.store_name.unique()
    keyword_count_dfs = [
        keyword_count_store_key_df(df, key, store, join_keywords)
        for store in store_names
    ]
    store_sizes = [len(df[df.store_name == store]) for store in store_names]

    counts, pcts, _ = generate_color_map(
        keyword_count_dfs, store_sizes, store_names, how
    )

    return counts, pcts

def df_to_colortable(df, key, how='store_size', join_keywords=False):
    counts, pcts = df_to_colortable_data(
        df, key, how, join_keywords
    )
    color_table = colorize_df(pcts * 100)
    return counts, pcts, color_table



FLOWER_LIST = ['hydrangea',
 'rose',
 'orchid',
 'peony',
 'ranunculus',
 'tulip',
 'dahlia',
 'magnolia',
 'calla lily',
 'cherryblossom',
 'lily',
 'anemone',
 'daisy',
 'delphinium',
 'amaryllis',
 'lilac',
 'chrysanthemum',
 'carnation',
 'hellebore',
 'poppy',
 'protea',
 'anthurium',
 'allium',
 'poinsettia',
 'sunflower',
 'zinnia',
 'lavender',
 'gladiolus',
 'viburnum',
 'snapdragon',
 'hyacinth',
 'jasmine',
 'geranium',
 'narcissus',
 'iris',
 'bougainvillea',
 'pansy',
 'sweet pea',
 'bird of paradise',
 'gardenia',
 'lily of the valley',
 "baby's breath",
 'lisianthus',
 'camellia',
 'dogwood',
 'petunia',
 'bromeliad',
 'heliconia',
 'waxflower',
 'freesia',
 'grape hyacinth',
 'pincushion protea',
 'forsythia',
 'cosmos',
 'bells of ireland',
 'hibiscus',
 'amaranthus',
 'yarrow',
 'succulent',
 'helleborus',
 'mimosa',
 'astilbe',
 'flower',
 'primrose',
 'tuberose',
 'agapanthus',
 'cornflower',
 'gerbera',
 'stock',
 'wisteria',
 'artichoke',
 'azalea',
 'scabiosa',
 'african violet',
 "queen anne's lace",
 'roses',
 'gloriosa lily',
 'kalanchoe',
 'cyclamen',
 'marigold',
 'pussy willow',
 'salvia',
 'thistle',
 'violet',
 'water lily',
 'billy balls',
 'cockscomb',
 'craspedia',
 'foxglove',
 'kangaroo paw',
 'ornamental cabbage',
 'unknown',
 'aloe',
 'alstroemeria',
 'artichoke flower',
 'banksia',
 'begonia',
 'bell of ireland',
 'bellflower',
 'billy buttons',
 'blossom',
 'california poppy',
 'campanula',
 'clematis',
 'crocus',
 'dianthus',
 'eucalyptus',
 'forget me not',
 'fountain grass',
 'fritillaria',
 'gerbera daisy',
 'ginger',
 'gypsophila',
 'millet',
 'other',
 'peace lily',
 'phlox',
 'plumeria',
 'pompom flower',
 'statice',
 'strawberry flower',
 'veronica',
 'wildflower']