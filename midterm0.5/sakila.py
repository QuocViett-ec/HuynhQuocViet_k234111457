
import os, warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")  
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# ========== Imports ==========
from flask import Flask, render_template_string, request, send_file, abort, redirect, url_for
import io
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Connector nội bộ của bạn
from connectors.connector import Connector

app = Flask(__name__)

# ==============================
# 0) Helpers: Kết nối & Query
# ==============================
def get_conn(db="sakila") -> Connector:
    conn = Connector(database=db)
    conn.connect()
    return conn

def df_query(sql: str) -> pd.DataFrame:
    conn = get_conn()
    return conn.queryDataset(sql)

def sql_escape(value: str) -> str:
    return value.replace("'", "''") if isinstance(value, str) else value

# ======================================================
# I) CUSTOMERS BY FILM
# ======================================================
def films_overview() -> pd.DataFrame:
    sql = """
    SELECT
        f.film_id,
        f.title,
        COUNT(r.rental_id) AS rentals_count,
        COUNT(DISTINCT r.customer_id) AS unique_customers
    FROM film f
    JOIN inventory i  ON i.film_id = f.film_id
    LEFT JOIN rental r ON r.inventory_id = i.inventory_id
    GROUP BY f.film_id, f.title
    ORDER BY rentals_count DESC, f.title;
    """
    return df_query(sql)

def customers_by_film(film_id: int = None, film_title: str = None) -> pd.DataFrame:
    where = ""
    if film_id is not None:
        where = f"WHERE f.film_id = {int(film_id)}"
    elif film_title:
        film_title = sql_escape(film_title)
        where = f"WHERE f.title = '{film_title}'"

    sql = f"""
    SELECT
        f.film_id,
        f.title,
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        COUNT(r.rental_id) AS times_rented
    FROM film f
    JOIN inventory i ON i.film_id = f.film_id
    JOIN rental r    ON r.inventory_id = i.inventory_id
    JOIN customer c  ON c.customer_id = r.customer_id
    {where}
    GROUP BY f.film_id, f.title, c.customer_id
    ORDER BY times_rented DESC, customer_name;
    """
    return df_query(sql)

# ======================================================
# II) CUSTOMERS BY CATEGORY
# ======================================================
def categories_overview() -> pd.DataFrame:
    sql = """
    SELECT
        cat.category_id,
        cat.name AS category_name,
        COUNT(r.rental_id) AS rentals_count,
        COUNT(DISTINCT r.customer_id) AS unique_customers
    FROM category cat
    JOIN film_category fc ON fc.category_id = cat.category_id
    JOIN film f           ON f.film_id      = fc.film_id
    JOIN inventory i      ON i.film_id      = f.film_id
    LEFT JOIN rental r    ON r.inventory_id = i.inventory_id
    GROUP BY cat.category_id, cat.name
    ORDER BY rentals_count DESC, category_name;
    """
    return df_query(sql)

def customers_by_category(category_id: int = None, category_name: str = None) -> pd.DataFrame:
    where = ""
    if category_id is not None:
        where = f"WHERE cat.category_id = {int(category_id)}"
    elif category_name:
        category_name = sql_escape(category_name)
        where = f"WHERE cat.name = '{category_name}'"

    sql = f"""
    SELECT
        cat.category_id,
        cat.name AS category_name,
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        COUNT(r.rental_id) AS times_rented
    FROM category cat
    JOIN film_category fc ON fc.category_id = cat.category_id
    JOIN film f           ON f.film_id      = fc.film_id
    JOIN inventory i      ON i.film_id      = f.film_id
    JOIN rental r         ON r.inventory_id = i.inventory_id
    JOIN customer c       ON c.customer_id  = r.customer_id
    {where}
    GROUP BY cat.category_id, cat.name, c.customer_id
    ORDER BY times_rented DESC, customer_name;
    """
    return df_query(sql)

# ======================================================
# III) FEATURES & KMEANS
# ======================================================
def load_customer_features() -> pd.DataFrame:
    # Tạo đặc trưng mô tả "mức độ quan tâm Film/Inventory"
    sql = """
    WITH base AS (
        SELECT
            c.customer_id,
            CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
            r.rental_id,
            r.rental_date,
            r.return_date,
            i.inventory_id,
            i.store_id,
            f.film_id,
            f.length           AS film_length,
            f.rental_duration  AS rental_duration,
            f.rental_rate      AS rental_rate,
            f.replacement_cost AS replacement_cost
        FROM customer c
        LEFT JOIN rental r     ON r.customer_id = c.customer_id
        LEFT JOIN inventory i  ON i.inventory_id = r.inventory_id
        LEFT JOIN film f       ON f.film_id = i.film_id
    ),
    cat_map AS (
        SELECT DISTINCT i.inventory_id, fc.category_id
        FROM inventory i
        JOIN film_category fc ON fc.film_id = i.film_id
    ),
    joined AS (
        SELECT b.*, cm.category_id
        FROM base b
        LEFT JOIN cat_map cm ON cm.inventory_id = b.inventory_id
    ),
    agg AS (
        SELECT
            customer_id,
            MIN(customer_name) AS customer_name,
            COUNT(rental_id) AS rentals_total,
            COUNT(DISTINCT film_id) AS distinct_films,
            COUNT(DISTINCT category_id) AS distinct_categories,
            AVG(film_length) AS avg_film_length,
            AVG(rental_duration) AS avg_rental_duration_days,
            AVG(rental_rate) AS avg_rental_rate,
            AVG(replacement_cost) AS avg_replacement_cost,
            SUM(CASE WHEN store_id = 1 THEN 1 ELSE 0 END) AS rentals_store_1,
            SUM(CASE WHEN store_id = 2 THEN 1 ELSE 0 END) AS rentals_store_2,
            DATEDIFF(CURDATE(), MAX(rental_date)) AS recency_days,
            SUM(rental_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)) AS last_30d_count,
            SUM(rental_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)) AS last_90d_count
        FROM joined
        GROUP BY customer_id
    )
    SELECT * FROM agg;
    """
    df = df_query(sql)
    num_cols = [c for c in df.columns if c not in ("customer_id", "customer_name")]
    df[num_cols] = df[num_cols].fillna(0)
    return df

def kmeans_customers(df_features: pd.DataFrame, k: int = 5, random_state: int = 42):
    num_cols = [c for c in df_features.columns if c not in ("customer_id", "customer_name")]
    X = df_features[num_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, init='k-means++', max_iter=500, random_state=random_state)
    labels = km.fit_predict(X_scaled)

    result = df_features.copy()
    result["cluster"] = labels
    return result, km, scaler, num_cols

def cluster_stats(df: pd.DataFrame) -> list:
    stats = []
    for cid in sorted(df["cluster"].unique()):
        g = df[df["cluster"] == cid]
        stats.append({
            "cluster": int(cid),
            "count": int(len(g)),
            "rentals_mean": float(np.round(g["rentals_total"].mean(), 2)),
            "films_mean": float(np.round(g["distinct_films"].mean(), 2)),
            "cats_mean": float(np.round(g["distinct_categories"].mean(), 2)),
            "recency_mean": float(np.round(g["recency_days"].mean(), 2)),
        })
    return stats

# Plotly nền sáng, màu tương phản
def plot_3d_html(df: pd.DataFrame) -> str:
    d = df.copy()
    d["cluster"] = d["cluster"].astype(str)
    fig = px.scatter_3d(
        d, x="rentals_total", y="distinct_films", z="distinct_categories",
        color="cluster",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data=["customer_name", "recency_days"],
        title="Customer Segments – 3D View (Rentals • Distinct Films • Distinct Categories)",
        template="plotly_white",
    )
    fig.update_layout(
        legend_title_text="Cluster",
        margin=dict(l=0, r=0, b=0, t=40),
        height=550,
        scene=dict(
            bgcolor="white",
            xaxis=dict(title="Rentals", color="#111827", gridcolor="#e5e7eb"),
            yaxis=dict(title="Distinct Films", color="#111827", gridcolor="#e5e7eb"),
            zaxis=dict(title="Distinct Categories", color="#111827", gridcolor="#e5e7eb"),
        ),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def plot_2d_html(df: pd.DataFrame, title="Rentals × Distinct Films (colored by cluster)") -> str:
    d = df.copy()
    d["cluster"] = d["cluster"].astype(str)
    fig = px.scatter(
        d, x="rentals_total", y="distinct_films", color="cluster",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data=["customer_name", "recency_days", "distinct_categories"],
        title=title, template="plotly_white",
    )
    fig.update_layout(legend_title_text="Cluster", margin=dict(l=0, r=0, b=0, t=40), height=420)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

# Tải dữ liệu & chạy KMeans lần đầu
K_DEFAULT = int(os.getenv("KMEANS_K", "5"))
_features = load_customer_features()
df_clusters, km_model, scaler_model, used_cols = kmeans_customers(_features, k=K_DEFAULT)

# ======================================================
# IV) HTML FRAGMENTS: CSS chung (sáng, tương phản cao)
# ======================================================
BASE_CSS = """
<style>
  :root{
    --brand1:#2563eb;   /* blue */
    --brand2:#22c55e;   /* green */
    --ink:#111827;      /* text chính */
    --muted:#6b7280;    /* text phụ */
    --card:#ffffff;     /* nền thẻ */
    --bg:#f6f8fb;       /* nền trang */
    --border:#e5e7eb;
  }
  body{ background: var(--bg); color: var(--ink); }
  .hero{
    background: linear-gradient(135deg, #e0f2fe 0%, #dcfce7 100%);
    border-radius: 18px; padding: 24px;
    border: 1px solid var(--border);
    box-shadow: 0 10px 24px rgba(37,99,235,.12);
  }
  .card{ border:1px solid var(--border); border-radius: 16px; background: var(--card); }
  .card-glass{ background: var(--card); }
  .metric-number{ font-size: 26px; font-weight: 800; color: var(--ink); }
  .metric-label{ color: var(--muted); font-weight: 600; }
  h5, h4, h3 { color: var(--ink); }
  a.btn{ border-radius: 12px; }
</style>
"""

# ======================================================
# V) ROUTES — Dashboard, Films, Categories, Clusters
# ======================================================
@app.route("/")
def root():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    global df_clusters, K_DEFAULT, _features
    # điều chỉnh K
    k_param = request.args.get("k", "").strip()
    if k_param.isdigit() and int(k_param) > 0:
        K_DEFAULT = int(k_param)
        df_clusters, _, _, _ = kmeans_customers(_features, k=K_DEFAULT)

    stats = cluster_stats(df_clusters)
    graph3d = plot_3d_html(df_clusters)
    graph2d = plot_2d_html(df_clusters)

    html = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sakila • Customer Segmentation Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  """ + BASE_CSS + """
</head>
<body>
  <div class="container py-4">

    <!-- Hero -->
    <div class="hero mb-4">
      <div class="d-flex align-items-center justify-content-between flex-wrap gap-3">
        <div>
          <h2 class="m-0">Customer Segmentation Dashboard (sakila)</h2>
          <div class="text-muted">KMeans • Số cụm hiện tại: <b>{{ K_DEFAULT }}</b></div>
        </div>
        <form class="d-flex" method="get" action="{{ url_for('dashboard') }}">
          <input class="form-control me-2" style="max-width:140px" type="number" min="1" name="k" placeholder="Đổi K..." value="{{ K_DEFAULT }}">
          <button class="btn btn-primary" type="submit">Áp dụng</button>
        </form>
      </div>
      <div class="mt-3 d-flex gap-2">
        <a class="btn btn-outline-primary" href="{{ url_for('films_page') }}"> Films</a>
        <a class="btn btn-outline-success" href="{{ url_for('categories_page') }}">🏷 Categories</a>
      </div>
    </div>

    <!-- Cards thống kê cụm -->
    <div class="row g-3 mb-4">
      {% for s in stats %}
      <div class="col-12 col-sm-6 col-lg-4">
        <div class="card card-glass h-100">
          <div class="card-body">
            <div class="d-flex justify-content-between align-items-start">
              <div>
                <div class="metric-label">Cluster</div>
                <div class="metric-number">#{{ s.cluster + 1 }}</div>
              </div>
              <a class="btn btn-outline-primary btn-sm" href="{{ url_for('cluster_page', cluster_id=s.cluster) }}">Xem chi tiết</a>
            </div>
            <hr class="border-secondary opacity-25">
            <div class="row text-center">
              <div class="col">
                <div class="metric-label">Khách hàng</div>
                <div class="metric-number">{{ s.count }}</div>
              </div>
              <div class="col">
                <div class="metric-label">Rentals TB</div>
                <div class="metric-number">{{ "%.1f"|format(s.rentals_mean) }}</div>
              </div>
              <div class="col">
                <div class="metric-label">Films TB</div>
                <div class="metric-number">{{ "%.1f"|format(s.films_mean) }}</div>
              </div>
              <div class="col">
                <div class="metric-label">Recency TB</div>
                <div class="metric-number">{{ "%.1f"|format(s.recency_mean) }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      {% endfor %}
    </div>

    <!-- Charts -->
    <div class="row g-3">
      <div class="col-12 col-lg-6">
        <div class="card card-glass shadow-sm">
          <div class="card-body">
            <h5 class="mb-3">3D Scatter (Rentals • Distinct Films • Distinct Categories)</h5>
            {{ graph3d|safe }}
          </div>
        </div>
      </div>
      <div class="col-12 col-lg-6">
        <div class="card card-glass shadow-sm">
          <div class="card-body">
            <h5 class="mb-3">2D Scatter (Rentals × Distinct Films)</h5>
            {{ graph2d|safe }}
          </div>
        </div>
      </div>
    </div>

    <div class="text-center text-muted mt-4">
      Made By QuocViett
    </div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    return render_template_string(html, stats=stats, K_DEFAULT=K_DEFAULT,
                                  graph3d=graph3d, graph2d=graph2d)

# --------- Films (overview + detail + Excel) ----------
@app.route("/films")
def films_page():
    df = films_overview()
    table_html = df.to_html(index=False, classes="table table-striped table-hover table-bordered align-middle",
                            border=0, justify="center")

    html = """
<!doctype html>
<html lang="vi"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Films • Overview</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  """ + BASE_CSS + """
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/dataTables.bootstrap5.min.css">
</head>
<body>
<div class="container py-4">
  <div class="hero mb-4 d-flex align-items-center justify-content-between">
    <h3 class="m-0"> Films • Tổng quan</h3>
    <a class="btn btn-outline-secondary" href="{{ url_for('dashboard') }}">← Dashboard</a>
  </div>

  <div class="card shadow-sm">
    <div class="card-body">
      <div class="table-responsive">
        """ + table_html + """
      </div>
      <small class="text-muted">Nhấn vào tiêu đề cột để sắp xếp. Dùng ô tìm kiếm của DataTables.</small>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/dataTables.bootstrap5.min.js"></script>
<script>
  $(function(){ 
    const t = $('table').DataTable({pageLength:10, lengthMenu:[5,10,25,50,100]});
    // Link tiêu đề film sang trang chi tiết
    const idxTitle = $('table thead th').filter(function(){return $(this).text().trim().toLowerCase()==='title';}).index();
    const idxId    = $('table thead th').filter(function(){return $(this).text().trim().toLowerCase()==='film_id';}).index();
    $('table tbody tr').each(function(){
      const $tds = $(this).find('td');
      const id = $tds.eq(idxId).text().trim();
      const title = $tds.eq(idxTitle).text().trim();
      $tds.eq(idxTitle).html('<a href="/film/'+id+'" class="link-primary">'+title+'</a>');
    });
  });
</script>
</body></html>
"""
    return render_template_string(html)

@app.route("/film/<int:film_id>")
def film_detail(film_id: int):
    df = customers_by_film(film_id=film_id)
    if df.empty:
        abort(404, description="Không có dữ liệu film này.")
    title = df.iloc[0]["title"]
    table_html = df[["customer_id","customer_name","times_rented"]].to_html(index=False,
                    classes="table table-striped table-hover table-bordered align-middle",
                    border=0, justify="center")

    html = f"""
<!doctype html>
<html lang="vi"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Film • {title}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  """ + BASE_CSS + f"""
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/dataTables.bootstrap5.min.css">
</head>
<body>
<div class="container py-4">
  <div class="hero mb-4 d-flex justify-content-between align-items-center">
    <div>
      <h3 class="m-0"> Film: {title}</h3>
      <div class="text-muted">Danh sách khách hàng đã thuê (không trùng + số lần)</div>
    </div>
    <div class="d-flex gap-2">
      <a class="btn btn-success" href="/film/{film_id}/excel">Tải Excel</a>
      <a class="btn btn-outline-secondary" href="/films">← Films</a>
    </div>
  </div>

  <div class="card shadow-sm">
    <div class="card-body">
      <div class="table-responsive">{table_html}</div>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/dataTables.bootstrap5.min.js"></script>
<script>$(function(){{ $('table').DataTable({{pageLength:10}}); }});</script>
</body></html>
"""
    return render_template_string(html)

@app.route("/film/<int:film_id>/excel")
def film_excel(film_id: int):
    df = customers_by_film(film_id=film_id)
    if df.empty: abort(404, description="Không có dữ liệu film này.")
    title = df.iloc[0]["title"]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Customers")
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"film_{film_id}_{title.replace(' ','_')}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------- Categories (overview + detail + Excel) ----------
@app.route("/categories")
def categories_page():
    df = categories_overview()
    table_html = df.to_html(index=False, classes="table table-striped table-hover table-bordered align-middle",
                            border=0, justify="center")

    html = """
<!doctype html>
<html lang="vi"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Categories • Overview</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  """ + BASE_CSS + """
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/dataTables.bootstrap5.min.css">
</head>
<body>
<div class="container py-4">
  <div class="hero mb-4 d-flex align-items-center justify-content-between">
    <h3 class="m-0">🏷 Categories • Tổng quan</h3>
    <a class="btn btn-outline-secondary" href="{{ url_for('dashboard') }}">← Dashboard</a>
  </div>

  <div class="card shadow-sm">
    <div class="card-body">
      <div class="table-responsive">
        """ + table_html + """
      </div>
      <small class="text-muted">Nhấn vào tên Category để xem chi tiết.</small>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/dataTables.bootstrap5.min.js"></script>
<script>
  $(function(){
    $('table').DataTable({pageLength:10, lengthMenu:[5,10,25,50,100]});
    const idxName = $('table thead th').filter(function(){return $(this).text().trim().toLowerCase()==='category_name';}).index();
    const idxId   = $('table thead th').filter(function(){return $(this).text().trim().toLowerCase()==='category_id';}).index();
    $('table tbody tr').each(function(){
      const $tds = $(this).find('td');
      const id = $tds.eq(idxId).text().trim();
      const name = $tds.eq(idxName).text().trim();
      $tds.eq(idxName).html('<a href="/category/'+id+'" class="link-primary">'+name+'</a>');
    });
  });
</script>
</body></html>
"""
    return render_template_string(html)

@app.route("/category/<int:category_id>")
def category_detail(category_id: int):
    df = customers_by_category(category_id=category_id)
    if df.empty:
        abort(404, description="Không có dữ liệu category này.")
    name = df.iloc[0]["category_name"]
    table_html = df[["customer_id","customer_name","times_rented"]].to_html(index=False,
                    classes="table table-striped table-hover table-bordered align-middle",
                    border=0, justify="center")

    html = f"""
<!doctype html>
<html lang="vi"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Category • {name}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  """ + BASE_CSS + f"""
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/dataTables.bootstrap5.min.css">
</head>
<body>
<div class="container py-4">
  <div class="hero mb-4 d-flex justify-content-between align-items-center">
    <div>
      <h3 class="m-0">🏷Category: {name}</h3>
      <div class="text-muted">Danh sách khách hàng đã thuê (không trùng + số lần)</div>
    </div>
    <div class="d-flex gap-2">
      <a class="btn btn-success" href="/category/{category_id}/excel">Tải Excel</a>
      <a class="btn btn-outline-secondary" href="/categories">← Categories</a>
    </div>
  </div>

  <div class="card shadow-sm">
    <div class="card-body">
      <div class="table-responsive">{table_html}</div>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/dataTables.bootstrap5.min.js"></script>
<script>$(function(){{ $('table').DataTable({{pageLength:10}}); }});</script>
</body></html>
"""
    return render_template_string(html)

@app.route("/category/<int:category_id>/excel")
def category_excel(category_id: int):
    df = customers_by_category(category_id=category_id)
    if df.empty: abort(404, description="Không có dữ liệu category này.")
    name = df.iloc[0]["category_name"]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Customers")
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"category_{category_id}_{name}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------- Clusters: list + detail + Excel ----------
@app.route("/cluster/<int:cluster_id>")
def cluster_page(cluster_id: int):
    if 'cluster' not in df_clusters.columns:
        return "<h3>Dữ liệu chưa có cột 'cluster'.</h3>"
    df = df_clusters[df_clusters["cluster"] == cluster_id].copy()
    if df.empty:
        abort(404, description=f"Không tìm thấy Cluster {cluster_id+1}.")
    keep = ["customer_id","customer_name","rentals_total","distinct_films",
            "distinct_categories","recency_days","cluster"]
    df = df[keep]
    table_html = df.to_html(index=False, classes="table table-striped table-hover table-bordered align-middle",
                            border=0, justify="center")

    html = f"""
<!doctype html>
<html lang="vi"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cluster #{cluster_id+1}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  """ + BASE_CSS + f"""
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/dataTables.bootstrap5.min.css">
</head>
<body>
<div class="container py-4">
  <div class="hero mb-4 d-flex align-items-center justify-content-between">
    <h3 class="m-0">Cluster # {cluster_id+1}</h3>
    <div class="d-flex gap-2">
      <a class="btn btn-success" href="/cluster/{cluster_id}/excel">Tải Excel</a>
      <a class="btn btn-outline-secondary" href="/">← Dashboard</a>
    </div>
  </div>

  <div class="card shadow-sm">
    <div class="card-body">
      <div class="table-responsive">{table_html}</div>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/dataTables.bootstrap5.min.js"></script>
<script>$(function(){{ $('table').DataTable({{pageLength:10}}); }});</script>
</body></html>
"""
    return render_template_string(html)

@app.route("/cluster/<int:cluster_id>/excel")
def cluster_excel(cluster_id: int):
    df = df_clusters[df_clusters["cluster"] == cluster_id].copy()
    if df.empty: abort(404, description=f"Không tìm thấy Cluster {cluster_id+1}.")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name=f"Cluster_{cluster_id+1}")
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"cluster_{cluster_id+1}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
