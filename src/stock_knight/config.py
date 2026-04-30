from __future__ import annotations
 
from pathlib import Path
from typing import Literal
 
from pydantic import BaseModel, Field
 
 
class UniverseConfig(BaseModel):
     mode: Literal["nifty100_membership", "liquidity_rule"] = "liquidity_rule"
     # liquidity_rule settings
     lookback_days: int = 60
     preselect_top_n: int = 120
     final_top_n: int = 100
     min_price: float = 20.0
     min_traded_value_inr: float = 2_00_00_000.0  # 2 cr median traded value filter
 
 
class DataConfig(BaseModel):
     root_dir: Path = Field(default_factory=lambda: Path("data"))
     exchange: Literal["NSE"] = "NSE"
 
     # Storage layout (all under root_dir)
     ohlcv_dir: Path = Path("ohlcv")
     corporate_actions_dir: Path = Path("corporate_actions")
     universe_dir: Path = Path("universe")
 
 
class BrokerConfig(BaseModel):
     provider: str = "ZERODHA"  # placeholder
     api_key: str | None = None
     access_token: str | None = None
 
 
class ProjectConfig(BaseModel):
     data: DataConfig = Field(default_factory=DataConfig)
     universe: UniverseConfig = Field(default_factory=UniverseConfig)
     broker: BrokerConfig = Field(default_factory=BrokerConfig)
 
 
def default_config() -> ProjectConfig:
     return ProjectConfig()
 
