import os, config
import pandas as pd
import json, random
from tqdm import tqdm

df = pd.read_csv(
    os.path.join(config.df_path, "cv-other-dev.csv")
)

def create_json(data, percent):
    json_data = []
    i = 0
    while i < data.shape[0]:
        json_data.append({
            "idx": i,
            "path": os.path.join(config.df_path, data["filename"].iloc[i]),
            "text": data["text"].iloc[i]
        })
        i += 1
    random.shuffle(json_data)
    print(len(json_data))
    with open(os.path.join(config.json_data_path, "train.json"), "w") as f:
        total = len(json_data)
        i = 0
        while i < int(total - total / percent):
            r = json_data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i += 1
    print(">>> train.json created")
    with open(os.path.join(config.json_data_path, "test.json"), "w") as f:
        total = len(json_data)
        i = int(total - total / percent)
        while i < total:
            r = json_data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i += 1
    print(">>> test.json created")

if __name__ == "__main__":
    create_json(df, 30)

