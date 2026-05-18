from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, ValidationError

# ----------------------------
# Param schemas per query_id
# ----------------------------

class TopBatsmenStrikeRateParams(BaseModel):
    season: int = Field(..., ge=2008, le=2035)
    min_balls: int = Field(..., ge=1, le=2000)
    team: Optional[str] = None
    limit: int = Field(10, ge=1, le=50)

class TopRunScorersParams(BaseModel):
    season: int = Field(..., ge=2008, le=2035)
    team: Optional[str] = None
    limit: int = Field(10, ge=1, le=50)

# Allowlist mapping query_id -> schema
ALLOWLIST: Dict[str, Any] = {
    "top_batsmen_strike_rate": TopBatsmenStrikeRateParams,
    "top_run_scorers": TopRunScorersParams,
}

def validate_params(query_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if query_id not in ALLOWLIST:
        raise ValueError(f"query_id '{query_id}' is not allowlisted")

    schema = ALLOWLIST[query_id]
    obj = schema(**(params or {}))
    # Return dict with defaults applied and validated types
    return obj.model_dump()