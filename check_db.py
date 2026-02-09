import sqlite3

conn = sqlite3.connect('idealquant.db')
cur = conn.cursor()

# List tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)

# Check optimization results
for table in tables:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    print(f"  {table}: {count} rows")

# Show recent results
if 'optimization_results' in tables:
    cur.execute("SELECT id, process_id, strategy_index, method, net_profit, max_drawdown, profit_factor, trade_count FROM optimization_results ORDER BY id DESC LIMIT 5")
    print("\nRecent optimization results:")
    for r in cur.fetchall():
        print(f"  ID={r[0]}, process={r[1]}, strategy={r[2]}, method={r[3]}, profit={r[4]}, dd={r[5]}, pf={r[6]}, trades={r[7]}")

if 'group_optimization_results' in tables:
    cur.execute("SELECT id, process_id, strategy_index, phase, net_profit, max_drawdown, profit_factor, trade_count FROM group_optimization_results ORDER BY id DESC LIMIT 5")
    print("\nRecent group optimization results:")
    for r in cur.fetchall():
        print(f"  ID={r[0]}, process={r[1]}, strategy={r[2]}, phase={r[3]}, profit={r[4]}, dd={r[5]}, pf={r[6]}, trades={r[7]}")

conn.close()
