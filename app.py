import streamlit as st
import pandas as pd
import numpy as np
import random
import os
from PIL import Image, UnidentifiedImageError, ImageOps

FLOWER_LIST = [" ",'hydrangea',
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

COLOR_LIST = [" ",'white',
 'pink',
 'yellow',
 'red',
 'green',
 'purple',
 'orange',
 'cream',
 'blue',
 'peach',
 'magenta',
 'brown',
 'tan',
 'beige',
 'gold',
 'burgundy',
 'grey',
 'lavender',
 'violet']

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# --- Callbacks ---
def add_keyword_1(column):
    kw = st.session_state[f"new_keyword_{column}"].strip()
    if kw:
        st.session_state[f"keywords_{column}"].add(kw)
    st.session_state[f"new_keyword_{column}"] = ""  # clear input
    # ensure the new keyword appears selected
    st.session_state[f"selected_keywords_{column}"] = sorted(st.session_state[f"keywords_{column}"])

def on_multiselect_change(column):
    selected = set(st.session_state[f"keyword_multiselect_{column}"])
    # Remove anything unselected
    st.session_state[f"keywords_{column}"].intersection_update(selected)
    # Sync selected list
    st.session_state[f"selected_keywords_{column}"] = sorted(st.session_state[f"keywords_{column}"])


def filter_rows_by_all_keywords(df, col, keywords, all_keywords=False):
    if not keywords:
        return df

    keywords_set = set(k.lower() for k in keywords)

    def contains_keywords(cell):
        if cell is None:
            return False

        # Handle list-like cells
        if isinstance(cell, (list, tuple, set, np.ndarray)):
            cell_iterable = cell
        else:
            if pd.isna(cell):
                return False
            cell_iterable = [cell]

        cell_set = set(str(item).lower() for item in cell_iterable)

        if all_keywords:
            # EXACT match: all and only
            return cell_set == keywords_set
        else:
            # SUPERSET match: must contain all keywords (extras allowed)
            return bool(keywords_set & cell_set)
            #return keywords_set.issubset(cell_set)

    mask = df[col].apply(contains_keywords)
    return df[mask]


def filter_dataframe(df: pd.DataFrame, filter_columns = []) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns.
    Widgets are arranged in rows of 3 columns.
    """
    modify = st.checkbox("详细条件", value=True)

    if not modify:
        return df

    df = df.copy()

    # Convert datetimes into a standard format
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        
        if not filter_columns:
            filter_columns = df.columns
                
        to_filter_columns = st.multiselect("参数", filter_columns)

        # Arrange widgets in rows of 3
        for i in range(0, len(to_filter_columns), 3):
            row_cols = st.columns(3)
            for j, column in enumerate(to_filter_columns[i:i+3]):
                col_widget = row_cols[j]

                # Handle list/array/dict columns with keyword filtering
                if df[column].apply(lambda x: isinstance(x, (np.ndarray, list, dict))).any():
                    
                    col_widget.write(column)

                    if f"keywords_{column}" not in st.session_state:
                        st.session_state[f"keywords_{column}"] = set()
                    if f"selected_keywords_{column}" not in st.session_state:
                        st.session_state[f"selected_keywords_{column}"] = []

                    # add flower
                    if column == "花":

                        col_widget.selectbox('Flower',
                                             options=FLOWER_LIST[:15],
                                             key=f"new_keyword_{column}",
                                             on_change = add_keyword_1,
                                             args=(column,),
                            )
                        
                    elif column == "花色":

                        col_widget.selectbox('Color',
                                             options=COLOR_LIST,
                                             key=f"new_keyword_{column}",
                                             on_change = add_keyword_1,
                                             args=(column,),
                            )


                    else:
                        col_widget.text_input(
                        f"{column}:",
                        key=f"new_keyword_{column}",
                        on_change=add_keyword_1,
                        args=(column,),
                    )

                    col_widget.multiselect(
                        "Keywords:",
                        options=sorted(st.session_state[f"keywords_{column}"]),
                        default=sorted(st.session_state[f"keywords_{column}"]),
                        key=f"keyword_multiselect_{column}",
                        on_change=on_multiselect_change,
                        args=(column,),
                    )

                    exact_match = col_widget.checkbox("Exact match",
                                            key=f"exact_match_{column}")
                    

                    curr_filter_list = st.session_state[f"keywords_{column}"]
                    
                    if exact_match:
                        df = filter_rows_by_all_keywords(df, column, curr_filter_list, all_keywords = True)
                    else:
                        df = filter_rows_by_all_keywords(df, column, curr_filter_list, all_keywords = False)
                        

                # Treat categorical columns
                elif is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = col_widget.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]

                # Numeric columns
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = col_widget.slider(
                        f"Values for {column}",
                        _min,
                        _max,
                        (_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]

                # Datetime columns
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = col_widget.date_input(
                        f"Values for {column}",
                        value=(df[column].min(), df[column].max()),
                    )
                    if len(user_date_input) == 2:
                        start_date, end_date = map(pd.to_datetime, user_date_input)
                        df = df.loc[df[column].between(start_date, end_date)]

                # Fallback: text/regex filtering
                else:
                    user_text_input = col_widget.text_input(f"Substring or regex in {column}")
                    if user_text_input:
                        df = df[df[column].str.contains(user_text_input, na=False)]

    return df


# --- Streamlit Setup ---
st.set_page_config(page_title="Image Keyword Filter", layout="wide")
st.title("🖼️ Flower Pictures")

# --- Load DataFrame ---
df = pd.read_parquet("america_flower_df_chinese.parquet")
filter_columns = ['花', '花色', '产品总类', '载具', '商家', '花种类数量']

trimmed_df = filter_dataframe(df, filter_columns) 

if len(trimmed_df) > 0:
    st.success(f"有{len(trimmed_df)}图片!")
else:
    st.info("No matching images found.")

# st.write(trimmed_df)

if st.button("🎨 加载图片"):
    # If more than 24 images, randomly select 24 rows
    if len(trimmed_df) > 200:
        trimmed_sample = trimmed_df.head(200)
    else:
        trimmed_sample = trimmed_df.copy()

    # Display images in 3-column grid
    
    def to_str(val):
        """Convert list/ndarray to readable comma-separated string."""
        if isinstance(val, (list, set, tuple)):
            return ", ".join(map(str, val))
        if isinstance(val, np.ndarray):
            return ", ".join(map(str, val.tolist()))
        return str(val)

    # FIXED_SIZE = (1200, 1200)  # width, height for all images
    grid_cols = st.columns(3)


    for idx, (_, row) in enumerate(trimmed_sample.iterrows()):
        with grid_cols[idx % 3]:
            img_path = row["图片链接"]

            try:

                url = to_str(row.get("产品链接", ""))
                title = to_str(row.get("产品", ""))

                st.image(img_path)
                st.caption(title)
                st.caption(url)

                st.markdown(
                    f"""
                    **花:** {to_str(row.get("花", ""))}  
                    **花色:** {to_str(row.get("花色", []))}  
                    **商家:** {to_str(row.get("商家", []))}  

                    """
                )

            except (FileNotFoundError, UnidentifiedImageError, OSError):
                st.warning(f"⚠️ Could not load image: {img_path}")


























    # for idx, (_, row) in enumerate(trimmed_sample.iterrows()):
    #     with grid_cols[idx % 3]:
    #         img_path = row["image_path"]
    #         img_path = "wreath&garland/" + img_path

    #         try:
    #             if not os.path.exists(img_path):
    #                 st.warning(f"⚠️ File not found: {img_path}")
    #                 continue

    #             img = Image.open(img_path)

    #             # Resize to fixed size with proper resampling
    #             img = ImageOps.fit(img, FIXED_SIZE, method=Image.Resampling.LANCZOS)

    #             st.image(img, use_container_width=True)
    #             st.caption(img_path)

    #             st.markdown(
    #                 f"""
    #                 **Subtype:** {to_str(row.get("subtype", ""))}  
    #                 **Leaves:** {to_str(row.get("leaves", []))}  
    #                 **Decorations:** {to_str(row.get("decorations", []))}  
    #                 **Colors:** {to_str(row.get("colors", []))}
    #                 """
    #             )
    #             st.divider()

    #         except (FileNotFoundError, UnidentifiedImageError, OSError):
    #             st.warning(f"⚠️ Could not load image: {img_path}")

    # # Display filtered DataFrame below
    # st.markdown("### 📊 Filtered Data")
    # st.dataframe(trimmed_df)
