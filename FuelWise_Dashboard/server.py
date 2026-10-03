import argparse,csv,io,json,sqlite3,threading,webbrowser
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from analytics import analyze,clean
from importer import prepare
BASE=Path(__file__).resolve().parent

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
        def send_bytes(self,data,content_type='application/json; charset=utf-8',filename=None):
            self.send_response(200);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            url=urlparse(self.path);q=parse_qs(url.query)
            try:
                if url.path=='/api/catalog':data={'meta':meta,'trips':trips}
                elif url.path=='/api/trip':
                    data=trip(int(q['id'][0]));data={k:v for k,v in data.items() if k!='raw'}
                elif url.path=='/api/raw':
                    raw=trip(int(q['id'][0]))['raw'];page=max(0,int(q.get('page',['0'])[0]));data={'rows':raw[page*100:(page+1)*100],'total':len(raw)}
                elif url.path=='/api/export':
                    raw=trip(int(q['id'][0]))['raw'];out=io.StringIO();fields=list(dict.fromkeys(k for r in raw for k in r));writer=csv.DictWriter(out,fields);writer.writeheader()
                    # Prevent spreadsheet formula execution when opening exported text.
                    for row in raw:writer.writerow({k:("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v) for k,v in row.items()})
                    self.send_bytes(out.getvalue().encode('utf-8-sig'),'text/csv; charset=utf-8','journey.csv');return
                elif url.path=='/api/health':data={'ok':True,'trips':len(trips)}
                elif url.path in ('/','/index.html','/app.js','/style.css','/polish.css'):
                    super().do_GET();return
                else:self.send_error(404);return
                self.send_bytes(json.dumps(clean(data),ensure_ascii=False,allow_nan=False).encode())
            except (KeyError,ValueError,IndexError):self.send_error(400,'Invalid request')
            except BrokenPipeError:pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    address=f'http://127.0.0.1:{args.port}';print(f'\nFuelWise 已啟動：{address}\n資料 {meta["rows"]:,} 筆 / {len(trips):,} 趟。按 Ctrl+C 結束。',flush=True)
    if not args.no_browser:threading.Timer(1,lambda:webbrowser.open(address)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
