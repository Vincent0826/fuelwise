"use client";
import {useEffect,useState} from "react";
type Trip={vehicle_label:string;journey_code:string;start_time:string;end_time:string;distance_km:number;fuel_used_l:number;duration_s:number;fuel_l_per_100km:number;engine_on_stop_estimate_s:number;data_source:string;limitations:string[]};
type Payload={status:string;trip:Trip;prediction:{status:string;reason:string}};
const base=process.env.NEXT_PUBLIC_API_BASE_URL??"http://localhost:8000";
export default function Page(){
 const [data,setData]=useState<Payload|null>(null);
 const [error,setError]=useState("");
 const [loading,setLoading]=useState(true);
 async function load(){
  setLoading(true);setError("");
  try{const r=await fetch(`${base}/api/v1/demo/trip`);if(!r.ok)throw new Error(`HTTP ${r.status}`);setData(await r.json());}
  catch{setError("無法取得分析資料。請確認 Python 後端已啟動，再重新載入。");}
  finally{setLoading(false);}
 }
 useEffect(()=>{void load();},[]);
 return <main><header><div className="brand">FuelWise <span>油耗領航員</span></div><span className="badge">本機開發骨架</span></header>
 <section className="heading"><p>單趟油耗分析</p><h1>先看清楚每一趟，再決定如何改善。</h1><p>目前展示已分析行程的固定摘要；尚未接入資料庫或訓練模型。</p></section>
 {loading&&<p role="status">正在讀取行程…</p>}
 {error&&<section role="alert"><p>{error}</p><button onClick={()=>void load()}>重新載入</button></section>}
 {!loading&&data&&<><section><h2>{data.trip.vehicle_label}</h2><p>{data.trip.start_time} — {data.trip.end_time}</p><p>行程 {data.trip.journey_code}</p></section>
 <div className="metrics">{[["行駛距離",`${data.trip.distance_km} km`],["耗油端點差",`${data.trip.fuel_used_l} L`],["行程油耗估計",`${data.trip.fuel_l_per_100km.toFixed(2)} L/100 km`],["停車引擎運轉估計",`${data.trip.engine_on_stop_estimate_s} 秒`]].map(([label,value])=><section key={label}><p>{label}</p><strong>{value}</strong></section>)}</div>
 <div className="columns"><section><h2>如何理解這些結果</h2><ul>{data.trip.limitations.map(x=><li key={x}>{x}</li>)}</ul></section><section><h2>油耗模型</h2><p>{data.prediction.reason}</p><p>AI 助理與改善任務將在分析介面確認後接入。</p></section></div></>}
 </main>;
}
