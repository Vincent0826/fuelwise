from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from .services.analysis import Sample,analyze
from fastapi import HTTPException

app=FastAPI(title="FuelWise API",version="0.1.0")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:3000"],allow_methods=["GET","POST"],allow_headers=["Content-Type"])

class SampleInput(BaseModel):
    timestamp: datetime
    mileage_km: float=Field(ge=0,allow_inf_nan=False)
    total_fuel_l: float=Field(ge=0,allow_inf_nan=False)
    can_speed_kmh: float=Field(ge=0,allow_inf_nan=False)
    rpm: float=Field(ge=0,allow_inf_nan=False)

class AnalysisInput(BaseModel):
    samples: list[SampleInput]=Field(min_length=2,max_length=20000)

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/api/v1/demo/trip")
def demo():
    return {"status":"demo_summary","trip":{
        "vehicle_label":"ABC-6325","journey_code":"250121000002",
        "start_time":"2025-01-21 08:16:25","end_time":"2025-01-21 08:45:35",
        "distance_km":12.4,"fuel_used_l":2.5,"duration_s":1750,
        "fuel_l_per_100km":100*2.5/12.4,"engine_on_stop_estimate_s":313,
        "data_source":"先前使用者提供的 190 筆資料之分析摘要，非即時重算",
        "limitations":["原始時間未註明時區，正式匯入前須確認。",
        "停車引擎運轉時間採前值保持估算，不能直接等同可避免的怠速。",
        "累積油量以 0.5 L 跳升，無法可靠分配片段耗油。",
        "缺少相似行程基準，尚不能判定本趟油耗異常。"]},
        "prediction":{"status":"not_ready","reason":"尚未訓練油耗模型，因此目前不提供預測數字。"}}

@app.post("/api/v1/analyses/preview")
def preview(body:AnalysisInput):
    try: return analyze([Sample(**s.model_dump()) for s in body.samples])
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc

@app.get("/api/v1/models/status")
def model_status():
    return {"status":"not_ready","active_model":None,"reason":"尚無經驗證並啟用的模型"}
