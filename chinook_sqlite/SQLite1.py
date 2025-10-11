import sqlite3
import pandas as pd


def customers_with_min_invoices(db_path, N):
    """
    Returns list of customers who have participated in >= N invoices.
    """
    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT c.CustomerId, c.FirstName, c.LastName, COUNT(i.InvoiceId) AS InvoiceCount
    FROM Customer c
    JOIN Invoice i ON c.CustomerId = i.CustomerId
    GROUP BY c.CustomerId, c.FirstName, c.LastName
    HAVING COUNT(i.InvoiceId) >= {N}
    ORDER BY InvoiceCount DESC;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df
def top_invoices_in_range(db_path, a, b, N):
    """
    Trả về TOP N Invoice có tổng giá trị trong khoảng [a, b], sắp xếp giảm dần theo tổng giá trị.
    """
    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT InvoiceId, CustomerId, Total
    FROM Invoice
    WHERE Total BETWEEN {a} AND {b}
    ORDER BY Total DESC
    LIMIT {N};
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def top_customers_by_invoice_count(db_path, N):
    """
    Trả về TOP N khách hàng có nhiều Invoice nhất.
    """
    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT c.CustomerId, c.FirstName, c.LastName, COUNT(i.InvoiceId) AS InvoiceCount
    FROM Customer c
    JOIN Invoice i ON c.CustomerId = i.CustomerId
    GROUP BY c.CustomerId, c.FirstName, c.LastName
    ORDER BY InvoiceCount DESC
    LIMIT {N};
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def top_customers_by_total_invoice_value(db_path, N):
    """
    Trả về TOP N khách hàng có tổng giá trị Invoice cao nhất.
    """
    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT c.CustomerId, c.FirstName, c.LastName, SUM(i.Total) AS TotalSpent
    FROM Customer c
    JOIN Invoice i ON c.CustomerId = i.CustomerId
    GROUP BY c.CustomerId, c.FirstName, c.LastName
    ORDER BY TotalSpent DESC
    LIMIT {N};
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
# Example usage
result = customers_with_min_invoices("../databases/Chinook_Sqlite.sqlite", 5)
print(result)
db_path = "../databases/Chinook_Sqlite.sqlite"

print("TOP Invoice trong khoảng [5, 20]:")
print(top_invoices_in_range(db_path, 5, 20, 5))

print("\nTOP khách hàng có nhiều Invoice nhất:")
print(top_customers_by_invoice_count(db_path, 5))

print("\nTOP khách hàng có tổng giá trị Invoice cao nhất:")
print(top_customers_by_total_invoice_value(db_path, 5))
