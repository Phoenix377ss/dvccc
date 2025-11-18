import os
from pathlib import Path

import pandas as pd
from sklearn import datasets

# Текущая рабочая директория (корень проекта)
if __name__ == "__main__":
    BASE_DIR = Path(os.getcwd())
    DATA_DIR = BASE_DIR / "data/initial_data" 

    DATA_DIR.mkdir(exist_ok=True)

    dataset = datasets.load_diabetes()

    features = pd.DataFrame(
        data=dataset.data,
        columns=[f"feat{x}" for x in range(dataset.data.shape[1])]
    )
    target = pd.DataFrame(data=dataset.target, columns=["target"])

    features.to_csv(DATA_DIR / "initial_data.csv", index=False)
    target.to_csv(DATA_DIR / "target.csv", index=False)
