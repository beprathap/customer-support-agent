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

        #get the dimension of the vectors
    dimension = vectors.shape[1]
    #create a FAISS index with L2 distance metric (cosine similarity)
    index = faiss.IndexFlatL2(dimension)
    #add the vectors to the index
    index.add(vectors)
    #save the index to a file
    faiss.write_index(index, index_file_path)
    print("FAISS index is created and vectors are added to the index.")

    return index