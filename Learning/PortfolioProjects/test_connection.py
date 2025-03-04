import pyodbc

# Use the public SQL Server address
server = 'sqldbserver.database.windows.net'  # Update this!
database = 'DB'
username = 'abc'
password = 'xyz'
driver = '{ODBC Driver 18 for SQL Server}'

# Connection string
conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("SELECT @@VERSION")  # Check SQL Server version
    row = cursor.fetchone()
    print(f"Connected to Azure SQL Database!\nSQL Server Version: {row[0]}")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"Connection failed: {e}")
