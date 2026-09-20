"""純計算層：不依賴 FastAPI、資料庫或語言模型。時間必须帶時區。"""
from dataclasses import dataclass
from datetime import datetime
from math import isfinite

@dataclass(frozen=True)
class Sample:
    timestamp: datetime
    mileage_km: float
    total_fuel_l: float
    can_speed_kmh: float
    rpm: float

def analyze(samples: list[Sample]) -> dict:
    if len(samples)<2:
        raise ValueError("至少需要兩筆紀錄")
    for s in samples:
        if s.timestamp.utcoffset() is None:
            raise ValueError("必須確認來源時區，時間不可缺少時區")
        if not all(isfinite(v) and v>=0 for v in (s.mileage_km,s.total_fuel_l,s.can_speed_kmh,s.rpm)):
            raise ValueError("數值必須有限且非負")
    duration=stop=0.0
    for a,b in zip(samples,samples[1:]):
        dt=(b.timestamp-a.timestamp).total_seconds()
        if dt<=0: raise ValueError("時間重複或倒序")
        if b.mileage_km<a.mileage_km or b.total_fuel_l<a.total_fuel_l:
            raise ValueError("計數器下降，須先檢查重置或資料異常")
        duration+=dt
        if a.can_speed_kmh==0 and a.rpm>0: stop+=dt
    distance=samples[-1].mileage_km-samples[0].mileage_km
    fuel=samples[-1].total_fuel_l-samples[0].total_fuel_l
    return {"distance_km":distance,"fuel_used_l":fuel,"duration_s":duration,
            "fuel_l_per_100km":100*fuel/distance if distance>0 else None,
            "engine_on_stop_estimate_s":stop,"method":"left_hold",
            "analysis_version":"0.1.0"}
