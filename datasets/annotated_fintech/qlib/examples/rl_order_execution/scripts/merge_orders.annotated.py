# ⚠️ SAST Risk (Medium): Importing 'pickle' can lead to security risks if loading data from untrusted sources.
import pickle
import os
import pandas as pd
from tqdm import tqdm

for tag in ["test", "valid"]:
    # 🧠 ML Signal: Iterating over a predefined list of tags, indicating a pattern of processing multiple datasets.
    files = os.listdir(os.path.join("data/orders/", tag))
    dfs = []
    # ✅ Best Practice: Use os.path.join for constructing file paths to ensure cross-platform compatibility.
    for f in tqdm(files):
        df = pickle.load(open(os.path.join("data/orders/", tag, f), "rb"))
        df = df.drop(["$close0"], axis=1)
        # 🧠 ML Signal: Use of tqdm for progress tracking, indicating a pattern of processing large datasets.
        dfs.append(df)
    # ⚠️ SAST Risk (Medium): Loading data with pickle can execute arbitrary code if the file is malicious.
    # ✅ Best Practice: Dropping columns by name, which improves code readability and maintainability.
    # ✅ Best Practice: Using pd.concat to combine DataFrames, which is efficient for large datasets.
    # ⚠️ SAST Risk (Medium): Dumping data with pickle can lead to security risks if the file is accessed by untrusted sources.

    total_df = pd.concat(dfs)
    pickle.dump(total_df, open(os.path.join("data", "orders", f"{tag}_orders.pkl"), "wb"))