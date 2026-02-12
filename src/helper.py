import faiss
import numpy as np

from typing import Tuple, List

import pandas as pd

from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI

API_KEY = 'YOUR KEY HERE'

def create_embeddings(df: pd.DataFrame, column_name: str, model: str) -> np.ndarray:
    """
    This function loads the OpenAI embedding model, encodes the text data in the specified column, 
    and returns a NumPy array of the embeddings.

    Args:
        df (pandas.DataFrame): The DataFrame containing the text data.
        column_name (str): The name of the column containing the text data.
        model (str): The name of the OpenAI embedding model.

    Returns:
        np.ndarray: A NumPy array containing the vector embeddings.
    """