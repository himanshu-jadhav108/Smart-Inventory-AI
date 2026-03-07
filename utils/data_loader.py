"""
Data loading and validation module
"""

import streamlit as st
import pandas as pd
from data.sample_generator import generate_sample_data


def load_and_validate_data(uploaded_file=None):
    """
    Load CSV file or generate sample data, then validate and clean it.

    Args:
        uploaded_file: Streamlit UploadedFile object or None

    Returns:
        Cleaned DataFrame or None on failure
    """
    if uploaded_file is not None:
        try:
            # Reset file pointer (important when re-uploading)
            uploaded_file.seek(0)

            # Check file size
            if uploaded_file.size == 0:
                st.error("❌ Uploaded file is empty.")
                return None

            # Read CSV
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("❌ Uploaded CSV contains no data.")
                return None

            # Normalize column names — strip whitespace and lowercase
            df.columns = df.columns.str.strip().str.lower()

            st.sidebar.success("✅ File uploaded successfully!")

        except pd.errors.EmptyDataError:
            st.error("❌ The file is empty or invalid CSV format.")
            return None

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None

    else:
        df = generate_sample_data()
        st.sidebar.info("📊 Using AI-generated sample data")

    # Validate required columns
    required_cols = ['date', 'product_id', 'units_sold']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        st.error(
            f"❌ Missing required columns: {', '.join(missing)}\n\n"
            "Required format: date, product_id, units_sold"
        )
        return None

    # Convert data types safely
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce').fillna(0)
    df = df.sort_values(['product_id', 'date']).reset_index(drop=True)

    return df
