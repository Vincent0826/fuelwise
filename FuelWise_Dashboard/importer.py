"""Streaming XLSX reader + atomic SQLite cache. XLS support via xlrd."""

import hashlib
import json
import posixpath
import re
import sqlite3
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from lxml import etree as E
from analytics import analyze, clean

VERSION = '4'
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def excel_rows(path):
    with open(path, 'rb') as f:
        magic = f.read(4)

    if magic[:2] != b'PK':
        book = pd.ExcelFile(path, engine='xlrd')
        for sheet in book.sheet_names:
            df = pd.read_excel(book, sheet_name=sheet, dtype=object)
            if {'journeyCode', 'time'}.issubset(df.columns):
                yield list(df.columns)
                for row in df.itertuples(index=False, name=None):
                    yield [None if pd.isna(value) else value for value in row]
                return
        raise ValueError('找不到含 journeyCode 與 time 的工作表')

    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as f:
                for _, el in E.iterparse(f, events=('end',), tag=NS + 'si'):
                    strings.append(''.join(el.itertext()))
                    el.clear()
                    while el.getprevious() is not None:
                        del el.getparent()[0]
        wb = E.fromstring(z.read('xl/workbook.xml'))
        rels = E.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        paths = {element.get('Id'): element.get('Target') for element in rels}
        props = wb.find(NS + 'workbookPr')
        epoch = (
            datetime(1904, 1, 1)
            if props is not None and props.get('date1904') in ('1', 'true')
            else datetime(1899, 12, 30)
        )

        for sheet in wb.find(NS + 'sheets'):
            relationship_id = sheet.get(
                '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
            )
            target = paths[relationship_id]
            target = (
                target.lstrip('/')
                if target.startswith('/')
                else posixpath.normpath('xl/' + target)
            )
            header = None
            chosen = False
            with z.open(target) as f:
                for _, row in E.iterparse(f, events=('end',), tag=NS + 'row'):
                    values = {}
                    for cell in row:
                        ref = re.sub(r'\d', '', cell.get('r', ''))
                        index = 0
                        for char in ref:
                            index = index * 26 + ord(char) - 64

                        value = cell.findtext(NS + 'v')
                        cell_type = cell.get('t')
                        if cell_type == 's':
                            value = strings[int(value)] if value is not None else None
                        elif cell_type == 'inlineStr':
                            value = ''.join(cell.find(NS + 'is').itertext())
                        elif cell_type == 'e':
                            value = None
                        elif value is not None:
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                        values[index - 1] = value

                    if header is None:
                        header = [
                            str(values.get(i) or '').strip()
                            for i in range(max(values, default=-1) + 1)
                        ]
                        chosen = {'journeyCode', 'time'}.issubset(header)
                        if not chosen:
                            break
                        yield header
                    else:
                        result = [values.get(i) for i in range(len(header))]
                        for i, column in enumerate(header):
                            is_date_column = column == 'time' or re.match(
                                r'event\[\d+\]\.(startTime|endTime)$', column
                            )
                            if is_date_column and isinstance(result[i], (int, float)):
                                result[i] = (
                                    epoch + timedelta(days=result[i])
                                ).isoformat(sep=' ')
                        yield result

                    row.clear()
                    while row.getprevious() is not None:
                        del row.getparent()[0]
            if chosen:
                return
        raise ValueError('找不到含 journeyCode 與 time 的工作表')


def identifier(v):
    if v is None or pd.isna(v):
        return ''
    if isinstance(v, (float, int)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def prepare(path, cache_dir):
    path = Path(path).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = path.stat()
    signature = f'{path}|{stat.st_size}|{stat.st_mtime_ns}|{VERSION}'
    final = cache_dir / (
        hashlib.sha256(signature.encode()).hexdigest()[:20] + '.sqlite'
    )
    if final.exists():
        return final

    temp = final.with_suffix('.building')
    temp.unlink(missing_ok=True)
    c = sqlite3.connect(temp)
    c.execute(
        'CREATE TABLE points(vehicle TEXT, journey TEXT, seq INTEGER, raw TEXT)'
    )
    c.execute(
        'CREATE TABLE trips(id INTEGER PRIMARY KEY, vehicle TEXT, journey TEXT, summary TEXT)'
    )
    c.execute('CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)')

    rows = excel_rows(path)
    headers = next(rows)
    batch = []
    skipped = 0
    count = 0
    print('首次讀取完整 Excel 並建立索引，請保留此視窗…',flush=True)
    for n, values in enumerate(rows, 2):
        raw = {
            column: clean(value)
            for column, value in zip(headers, values)
            if column and value is not None
        }
        vehicle = identifier(raw.get('enabledCode')) or identifier(raw.get('carNum'))
        journey = identifier(raw.get('journeyCode'))
        raw['source_row'] = n
        if not vehicle or not journey or not raw.get('time'):
            skipped += 1
            continue

        raw['journeyCode'] = journey
        batch.append(
            (
                vehicle,
                journey,
                n,
                json.dumps(
                    raw,
                    ensure_ascii=False,
                    default=str,
                    allow_nan=False,
                ),
            )
        )
        count += 1
        if len(batch) >= 5000:
            c.executemany('INSERT INTO points VALUES(?,?,?,?)', batch)
            batch = []
            if count % 50000 == 0:
                print(f'已讀 {count:,} 筆', flush=True)

    c.executemany('INSERT INTO points VALUES(?,?,?,?)', batch)
    c.execute('CREATE INDEX idx_points ON points(vehicle,journey,seq)')
    c.commit()
    keys = c.execute(
        'SELECT DISTINCT vehicle,journey FROM points'
    ).fetchall()
    print(f'建立 {len(keys):,} 趟行程摘要…', flush=True)
    failed = []
    for i, (vehicle, journey) in enumerate(keys):
        raw = [
            json.loads(row[0])
            for row in c.execute(
                'SELECT raw FROM points WHERE vehicle=? AND journey=? ORDER BY seq',
                (vehicle, journey),
            )
        ]
        try:
            summary = analyze(raw, False)
        except (ValueError, KeyError) as exc:
            failed.append(
                {'vehicle': vehicle, 'journey': journey, 'reason': str(exc)}
            )
            continue

        plate = next(
            (str(row['carNum']) for row in raw if row.get('carNum')),
            vehicle,
        )
        summary.update(vehicle=vehicle, journey=journey, plate=plate)
        c.execute(
            'INSERT INTO trips(vehicle,journey,summary) VALUES(?,?,?)',
            (vehicle, journey, json.dumps(summary, ensure_ascii=False)),
        )
        if (i + 1) % 500 == 0:
            print(f'已分析 {i + 1:,} 趟', flush=True)

    meta = {
        'source': path.name,
        'rows': count,
        'skipped_rows': skipped,
        'failed_trips': failed,
        'columns': headers,
        'cache_version': VERSION,
        'created': datetime.now().isoformat(),
    }
    c.execute(
        'INSERT INTO meta VALUES(?,?)',
        ('info', json.dumps(meta, ensure_ascii=False)),
    )
    c.commit()
    c.close()
    temp.replace(final)
    return final
