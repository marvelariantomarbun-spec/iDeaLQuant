import sqlite3, json

conn = sqlite3.connect('idealquant.db')
c = conn.cursor()

# List tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print("Tables:", tables)

# Check optimization_results or similar
for t in tables:
    c.execute(f"SELECT count(*) FROM [{t}]")
    cnt = c.fetchone()[0]
    print(f"  {t}: {cnt} rows")

# Find S2 results
for t in tables:
    c.execute(f"PRAGMA table_info([{t}])")
    cols = [r[1] for r in c.fetchall()]
    print(f"\n{t} columns: {cols}")
    
    # Get latest S2 result
    if 'strategy' in [c.lower() for c in cols] or 'strategy_index' in [c.lower() for c in cols]:
        strategy_col = 'strategy' if 'strategy' in cols else 'strategy_index'
        c.execute(f"SELECT * FROM [{t}] WHERE {strategy_col} = 1 ORDER BY rowid DESC LIMIT 5")
        rows = c.fetchall()
        for row in rows:
            print(dict(zip(cols, row)))
    elif 'params' in [c.lower() for c in cols]:
        c.execute(f"SELECT * FROM [{t}] ORDER BY rowid DESC LIMIT 3")
        rows = c.fetchall()
        for row in rows:
            print(dict(zip(cols, row)))

conn.close()
