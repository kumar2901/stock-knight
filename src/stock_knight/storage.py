from __future__ import annotations
 
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
 
import pandas as pd
 
 
@dataclass(frozen=True)
class DataPaths:
     root: Path
 
     @property
     def ohlcv(self) -> Path:
         return self.root / "ohlcv"
 
     @property
     def corporate_actions(self) -> Path:
         return self.root / "corporate_actions"
 
     @property
     def universe(self) -> Path:
         return self.root / "universe"
 
     @property
     def features(self) -> Path:
         return self.root / "features"
 
     @property
     def labels(self) -> Path:
         return self.root / "labels"
 
 
def ensure_dirs(*paths: Path) -> None:
     for p in paths:
         p.mkdir(parents=True, exist_ok=True)
 
 
def write_df_parquet(path: Path, df: pd.DataFrame) -> None:
     path.parent.mkdir(parents=True, exist_ok=True)
     df.to_parquet(path, index=True)
 
 
def read_df_parquet(path: Path) -> pd.DataFrame:
     return pd.read_parquet(path)
 
 
def write_json(path: Path, obj: Any) -> None:
     path.parent.mkdir(parents=True, exist_ok=True)
     path.write_text(json.dumps(obj, indent=2, default=str))
 
 
def read_json(path: Path) -> Any:
     return json.loads(path.read_text())
 
