import sqlite3
import pandas as pd

try:
    # Kết nối database
    sqliteConnection = sqlite3.connect('../databases/Chinook_Sqlite.sqlite')
    cursor = sqliteConnection.cursor()
    print("DB Init")

    # Câu query
    query = "SELECT * FROM InvoiceLine LIMIT 5;"
    cursor.execute(query)

    # Lấy kết quả và chuyển sang DataFrame
    records = cursor.fetchall()
    df = pd.DataFrame(records, columns=[desc[0] for desc in cursor.description])
    print(df)

    # Đóng cursor
    cursor.close()

except sqlite3.Error as error:
    print("Error occurred -", error)

finally:
    if sqliteConnection:
        sqliteConnection.close()
        print("SQLite Connection closed")
