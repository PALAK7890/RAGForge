import json
from pathlib import Path
from typing import Any, List, Union


class EvaluationDataset:

    def __init__(self, path: Union[str, Path]) -> None:

        self.path = Path(path)

    def load(self) -> List[Any]:

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Dataset must be a JSON array.")
            return data