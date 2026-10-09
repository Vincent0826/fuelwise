import argparse,csv,io,json,sqlite3,threading,webbrowser
import math,os,urllib.error,urllib.request
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from analytics import analyze,clean
from importer import prepare
from analysis.fuel_model_inference import (
    ModelUnavailableError,
    TripInferenceError,
    predict_trip,
)
from analysis.prepare_dashboard_model import prepare_dashboard_model
from analysis.trip_applicability import evaluate_applicability
BASE=Path(__file__).resolve().parent
AI_SENSOR_FIELDS=('speed','rpm','load','temp','battery','fuel_level','distance','used')
DEFAULT_GEMINI_MODEL='gemini-3.8-flash'

def finite_number(value):
    if isinstance(value,bool):return None
    try:number=float(value)
    except (TypeError,ValueError):return None
    return number if math.isfinite(number) else None

def sanitize_ai_context(context):
    if not isinstance(context,dict):raise ValueError('感測器資料格式錯誤')
    def sample(item,include_offset=False):
        if not isinstance(item,dict):return None
        result={key:finite_number(item.get(key)) for key in AI_SENSOR_FIELDS}
        if include_offset:result['offset_sec']=finite_number(item.get('offset_sec'))
        return result
    current=sample(context.get('current'),True)
    if current is None:raise ValueError('缺少目前感測器資料')
    recent=context.get('recent_points',[])
    if not isinstance(recent,list):raise ValueError('近期感測器資料格式錯誤')
    recent=[item for item in (sample(value,True) for value in recent[-20:]) if item is not None]
    summary=context.get('trip_summary',{})
    if not isinstance(summary,dict):raise ValueError('行程摘要格式錯誤')
    counts={}
    for key in ('observations','acceleration_events','braking_events'):
        value=finite_number(summary.get(key))
        counts[key]=int(value) if value is not None and 0<=value<=10000000 else 0
    stats={
        'max_acceleration_kmh_s':finite_number(summary.get('max_acceleration_kmh_s')),
        'max_braking_kmh_s':finite_number(summary.get('max_braking_kmh_s')),
    }
    events=summary.get('recent_events',[])
    if not isinstance(events,list):raise ValueError('加減速摘要格式錯誤')
    safe_events=[]
    for item in events[-10:]:
        if not isinstance(item,dict) or item.get('type') not in ('acceleration','braking'):continue
        safe_events.append({
            'type':item['type'],
            'start_offset_sec':finite_number(item.get('start_offset_sec')),
            'end_offset_sec':finite_number(item.get('end_offset_sec')),
            'max_kmh_s':finite_number(item.get('max_kmh_s')),
        })
    return {
        'threshold_kmh_s':finite_number(context.get('threshold_kmh_s')),
        'current':current,
        'recent_points':recent,
        'trip_summary':{**counts,**stats,'recent_events':safe_events},
    }

def ai_messages(message,history,sensor_context):
    if not isinstance(message,str) or not message.strip() or len(message)>1000:
        raise ValueError('請輸入 1,000 字以內的問題')
    if not isinstance(history,list):raise ValueError('對話紀錄格式錯誤')
    safe_history=[]
    for item in history[-12:]:
        if not isinstance(item,dict) or item.get('role') not in ('user','assistant'):continue
        content=item.get('content')
        if isinstance(content,str) and content.strip():
            safe_history.append({'role':item['role'],'content':content[:1000]})
    context=sanitize_ai_context(sensor_context)
    system_instruction=(
            '你是 FuelWise 的繁體中文駕駛行程分析助理。根據提供的匿名感測器資料回答駕駛問題，'
            '優先用簡短、清楚的自然語言，並指出數據依據。不要假設資料中沒有的車輛、路線或事件。'
            '加速度單位為 km/h/s；急加速或急減速的判定門檻是絕對值大於 3 km/h/s，且連續至少 3 筆有效觀測。'
            '這些資料是歷史行程重播，不是真正即時車聯網資料。資料不足時請明確說明，不要捏造數值。'
    )
    contents=[
        {'role':'user' if item['role']=='user' else 'model','parts':[{'text':item['content']}]}
        for item in safe_history
    ]
    contents.append({
        'role':'user',
        'parts':[{'text':(
            '以下是本次問題可參考的匿名感測器資料（不含車輛識別或 GPS）：\n'
            +json.dumps(context,ensure_ascii=False)
            +'\n\n駕駛問題：'+message.strip()
        )}],
    })
    return {
        'system_instruction':{'parts':[{'text':system_instruction}]},
        'contents':contents,
        'generation_config':{'temperature':0.3,'max_output_tokens':500},
    }

def gemini_model():
    return os.environ.get('GEMINI_MODEL',DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

def request_gemini(messages):
    api_key=os.environ.get('GEMINI_API_KEY','').strip()
    if not api_key:raise RuntimeError('尚未設定 GEMINI_API_KEY，請在啟動伺服器前設定環境變數。')
    model=gemini_model()
    if not model.startswith('gemini-') or not all(c.isalnum() or c in '._-' for c in model):
        raise ValueError('GEMINI_MODEL 設定無效')
    payload=json.dumps({
        'systemInstruction':messages['system_instruction'],
        'contents':messages['contents'],
        'generationConfig':messages['generation_config'],
    },ensure_ascii=False).encode()
    request=urllib.request.Request(
        f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
        data=payload,
        headers={'Content-Type':'application/json','X-goog-api-key':api_key},
        method='POST',
    )
    with urllib.request.urlopen(request,timeout=30) as response:
        result=json.loads(response.read())
    answer=''.join(
        part.get('text','')
        for part in result['candidates'][0]['content']['parts']
        if isinstance(part,dict)
    )
    if not isinstance(answer,str) or not answer.strip():raise ValueError('模型沒有回傳文字內容')
    return answer.strip()

def gemini_error_message(error,model,api_key=''):
    try:
        details=json.loads(error.read(8192))
        message=details.get('error',{}).get('message')
    except (json.JSONDecodeError,AttributeError,TypeError):
        message=None
    if not isinstance(message,str) or not message.strip():
        message='Gemini 未提供錯誤細節'
    if api_key:
        message=message.replace(api_key,'[REDACTED]')
    return f'Gemini API 請求失敗（HTTP {error.code}，模型 {model}）：{message[:600]}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--file',type=Path);p.add_argument('--port',type=int,default=8501);p.add_argument('--no-browser',action='store_true');p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    files=[BASE/'data.xlsx',BASE/'data.xls'];source=args.file or next((f for f in files if f.exists()),None)
    if source is None:raise SystemExit('請把 data.xlsx 或 data.xls 放在 server.py 同一資料夾，或使用 --file 路徑。')
    db=prepare(source,BASE/'.cache')
    if args.prepare_only:print(db);return
    def connection():return sqlite3.connect(f'file:{db.as_posix()}?mode=ro',uri=True)
    with connection() as c:
        meta=json.loads(c.execute('SELECT value FROM meta WHERE key="info"').fetchone()[0]);trips=[]
        for id_,s in c.execute('SELECT id,summary FROM trips'):
            s=json.loads(s);s['id']=id_;trips.append(s)
    model_build_error=None
    try:
        _,model_created=prepare_dashboard_model(db,meta)
        if model_created:print('行程油耗模型已建立，可於行程頁查看估計與解釋。',flush=True)
    except (FileNotFoundError,OSError,ValueError,RuntimeError,KeyError,ImportError) as exc:
        model_build_error=str(exc)
        print(f'行程油耗模型建立失敗；網站其他功能仍可使用：{exc}',flush=True)
    @lru_cache(maxsize=12)
    def trip(id_):
        with connection() as c:
            key=c.execute('SELECT vehicle,journey FROM trips WHERE id=?',(id_,)).fetchone()
            if key is None:raise KeyError('找不到行程')
            raw=[json.loads(r[0]) for r in c.execute('SELECT raw FROM points WHERE vehicle=? AND journey=? ORDER BY seq',key)]
        data=analyze(raw);data.update(vehicle=key[0],journey=key[1]);return data
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(BASE/'web'),**kw)
        def log_message(self,*a):pass
        def end_headers(self):
            if not self.path.startswith('/api/'):
                self.send_header('Cache-Control','no-store')
            super().end_headers()
        def send_bytes(self,data,content_type='application/json; charset=utf-8',filename=None):
            self.send_response(200);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
            self.end_headers();self.wfile.write(data)
        def send_json(self,data,status=200):
            body=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            url=urlparse(self.path);q=parse_qs(url.query)
            try:
                if url.path=='/api/catalog':data={'meta':meta,'trips':trips}
                elif url.path=='/api/trip':
                    data=trip(int(q['id'][0]));data={**{k:v for k,v in data.items() if k!='raw'},'applicability':evaluate_applicability(data)}
                elif url.path=='/api/fuel-model':
                    selected=trip(int(q['id'][0]))
                    try:data=predict_trip(selected,selected['vehicle'],selected['journey'])
                    except ModelUnavailableError as exc:
                        reason=str(exc)
                        if model_build_error:reason+=f'；本機自動訓練失敗：{model_build_error}'
                        self.send_json({'available':False,'reason':reason},503);return
                    except TripInferenceError as exc:
                        self.send_json({'available':False,'reason':str(exc)},422);return
                elif url.path=='/api/raw':
                    raw=trip(int(q['id'][0]))['raw'];page=max(0,int(q.get('page',['0'])[0]));data={'rows':raw[page*100:(page+1)*100],'total':len(raw)}
                elif url.path=='/api/export':
                    raw=trip(int(q['id'][0]))['raw'];out=io.StringIO();fields=list(dict.fromkeys(k for r in raw for k in r));writer=csv.DictWriter(out,fields);writer.writeheader()
                    # Prevent spreadsheet formula execution when opening exported text.
                    for row in raw:writer.writerow({k:("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v) for k,v in row.items()})
                    self.send_bytes(out.getvalue().encode('utf-8-sig'),'text/csv; charset=utf-8','journey.csv');return
                elif url.path=='/api/health':data={'ok':True,'trips':len(trips)}
                elif url.path=='/api/ai/status':
                    data={
                        'configured':bool(os.environ.get('GEMINI_API_KEY','').strip()),
                        'model':gemini_model(),
                    }
                elif url.path in ('/','/index.html','/app.js','/style.css','/polish.css'):
                    super().do_GET();return
                else:self.send_error(404);return
                self.send_bytes(json.dumps(clean(data),ensure_ascii=False,allow_nan=False).encode())
            except (KeyError,ValueError,IndexError):self.send_error(400,'Invalid request')
            except BrokenPipeError:pass
        def do_POST(self):
            if urlparse(self.path).path!='/api/ai/chat':
                self.send_error(404);return
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:
                self.send_json({'error':'請求格式錯誤'},400);return
            if length<=0 or length>131072:
                self.send_json({'error':'請求內容不得為空或超過 128 KB'},413 if length>131072 else 400);return
            try:body=json.loads(self.rfile.read(length))
            except (json.JSONDecodeError,UnicodeDecodeError):
                self.send_json({'error':'請求 JSON 格式錯誤'},400);return
            if not isinstance(body,dict):
                self.send_json({'error':'請求格式錯誤'},400);return
            try:messages=ai_messages(body.get('message'),body.get('history',[]),body.get('sensor_context'))
            except ValueError as exc:
                self.send_json({'error':str(exc)},400);return
            if not os.environ.get('GEMINI_API_KEY','').strip():
                self.send_json({'error':'尚未設定 GEMINI_API_KEY，請在啟動伺服器前設定環境變數。'},503);return
            try:answer=request_gemini(messages)
            except urllib.error.HTTPError as exc:
                model=gemini_model()
                api_key=os.environ.get('GEMINI_API_KEY','').strip()
                self.send_json({'error':gemini_error_message(exc,model,api_key)},502);return
            except (urllib.error.URLError,TimeoutError):
                self.send_json({'error':'連線 Gemini 失敗或逾時，請檢查網路後再試。'},502);return
            except (json.JSONDecodeError,KeyError,IndexError,TypeError,ValueError) as exc:
                self.send_json({'error':f'Gemini 回應格式錯誤：{exc}'},502);return
            except RuntimeError as exc:
                self.send_json({'error':str(exc)},503);return
            self.send_json({'answer':answer})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    address=f'http://127.0.0.1:{args.port}';print(f'\nFuelWise 已啟動：{address}\n資料 {meta["rows"]:,} 筆 / {len(trips):,} 趟。按 Ctrl+C 結束。',flush=True)
    if not args.no_browser:threading.Timer(1,lambda:webbrowser.open(address)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
