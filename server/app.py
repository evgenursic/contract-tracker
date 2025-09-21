import os
import sqlite3
import uuid
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional

import io

import pandas as pd
import base64
import urllib.parse
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

# Path to the database file (stored in the server directory)
DB_PATH = os.path.join(os.path.dirname(__file__), "contracts.db")


def get_db_connection():
    """Return a connection to the SQLite database.

    The connection uses row_factory to return rows as dictionaries for easier access.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the contracts table if it doesn't already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS contracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            value REAL NOT NULL,
            description TEXT,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def row_to_dict(row: sqlite3.Row) -> Dict:
    """Convert a sqlite3.Row to a regular dictionary."""
    return {key: row[key] for key in row.keys()}


def calculate_status(end_date: date) -> str:
    """Determine the textual status of a contract based on its end date.

    Args:
        end_date (date): The date when the contract ends.

    Returns:
        str: A status string: 'active', 'soon', or 'expired'.
    """
    today = date.today()
    if end_date < today:
        return "expired"
    days_left = (end_date - today).days
    if days_left <= 30:
        return "soon"
    return "active"


def calculate_urgency(end_date: date) -> str:
    """Determine the urgency level (kritično, opozorilo, informativno) based on days until expiry.

    Args:
        end_date (date): The contract's end date.

    Returns:
        str: One of 'kritično', 'opozorilo', 'informativno', or 'poteklo' (expired contracts).
    """
    today = date.today()
    delta = (end_date - today).days
    if delta < 0:
        return "poteklo"  # expired
    if delta <= 7:
        return "kritično"
    if delta <= 30:
        return "opozorilo"
    return "informativno"


def fetch_contracts(
    search: Optional[str] = None,
    status_filter: Optional[str] = None,
    urgency_filter: Optional[str] = None,
) -> List[Dict]:
    """Retrieve contracts from the database with optional filtering by name and status.

    Args:
        search (Optional[str]): A substring to search for within contract names.
        status_filter (Optional[str]): Filter by status ('active', 'soon', 'expired').
        urgency_filter (Optional[str]): Filter by urgency ('kritično', 'opozorilo', 'informativno', 'poteklo').

    Returns:
        List[Dict]: A list of contract records as dictionaries with computed fields for status and urgency.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM contracts"
    params: List = []
    filters: List[str] = []
    if search:
        filters.append("name LIKE ?")
        params.append(f"%{search}%")
    if filters:
        query += " WHERE " + " AND ".join(filters)
    query += " ORDER BY created_at DESC"
    rows = cursor.execute(query, params).fetchall()
    conn.close()
    contracts: List[Dict] = []
    for row in rows:
        record = row_to_dict(row)
        # Convert stored strings to date objects
        record['start_date'] = datetime.strptime(record['start_date'], "%Y-%m-%d").date()
        record['end_date'] = datetime.strptime(record['end_date'], "%Y-%m-%d").date()
        record['status'] = calculate_status(record['end_date'])
        record['urgency'] = calculate_urgency(record['end_date'])
        record['days_left'] = (record['end_date'] - date.today()).days
        contracts.append(record)
    # Apply status filter after computing statuses
    if status_filter:
        contracts = [c for c in contracts if c['status'] == status_filter]
    if urgency_filter:
        contracts = [c for c in contracts if c['urgency'] == urgency_filter]
    return contracts


def get_contract_by_id(contract_id: int) -> Optional[Dict]:
    """Retrieve a single contract by its ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    row = cursor.execute(
        "SELECT * FROM contracts WHERE id = ?", (contract_id,)
    ).fetchone()
    conn.close()
    if row:
        record = row_to_dict(row)
        record['start_date'] = datetime.strptime(record['start_date'], "%Y-%m-%d").date()
        record['end_date'] = datetime.strptime(record['end_date'], "%Y-%m-%d").date()
        return record
    return None


def insert_contract(name: str, start_date: date, end_date: date, value: float, description: str) -> None:
    """Insert a new contract record into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat(timespec='seconds')
    cursor.execute(
        """
        INSERT INTO contracts (name, start_date, end_date, value, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (name, start_date.isoformat(), end_date.isoformat(), value, description, now, now),
    )
    conn.commit()
    conn.close()


def update_contract(contract_id: int, name: str, start_date: date, end_date: date, value: float, description: str) -> None:
    """Update an existing contract."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat(timespec='seconds')
    cursor.execute(
        """
        UPDATE contracts
        SET name = ?, start_date = ?, end_date = ?, value = ?, description = ?, updated_at = ?
        WHERE id = ?
        """,
        (name, start_date.isoformat(), end_date.isoformat(), value, description, now, contract_id),
    )
    conn.commit()
    conn.close()


def delete_contract(contract_id: int) -> None:
    """Delete a contract by its ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM contracts WHERE id = ?", (contract_id,))
    conn.commit()
    conn.close()


def compute_dashboard_metrics() -> Dict:
    """Compute metrics for the dashboard: counts of total, active, soon expiring, expired and total value."""
    contracts = fetch_contracts()
    total = len(contracts)
    active = sum(1 for c in contracts if c['status'] == 'active')
    soon = sum(1 for c in contracts if c['status'] == 'soon')
    expired = sum(1 for c in contracts if c['status'] == 'expired')
    total_value = sum(c['value'] for c in contracts)
    # Sort by created_at descending and pick 5 latest
    latest = sorted(contracts, key=lambda x: x['id'], reverse=True)[:5]
    return {
        'total': total,
        'active': active,
        'soon': soon,
        'expired': expired,
        'total_value': total_value,
        'latest': latest,
    }


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return filename.lower().endswith((".xls", ".xlsx", ".csv"))


#############################
# FastAPI Application Setup #
#############################

app = FastAPI(title="Sledenje pogodb")

# Ensure database exists on startup
init_db()

# Configure templates and static files
BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static",
)

# In-memory storage for uploaded preview data
preview_store: Dict[str, List[Dict]] = {}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the dashboard showing key metrics and upcoming expiries."""
    metrics = compute_dashboard_metrics()
    # Identify contracts that expire soon (within next 30 days) for the alert list
    soon_contracts = [c for c in fetch_contracts() if c['status'] == 'soon']
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "metrics": metrics,
            "soon_contracts": soon_contracts,
        },
    )


@app.get("/contracts", response_class=HTMLResponse)
async def list_contracts(request: Request, search: str = "", status: str = ""):
    """List all contracts with optional search and status filtering."""
    status_filter = status if status in {"active", "soon", "expired"} else None
    contracts = fetch_contracts(search=search or None, status_filter=status_filter)
    return templates.TemplateResponse(
        "contracts.html",
        {
            "request": request,
            "contracts": contracts,
            "search": search,
            "status": status,
        },
    )


@app.get("/contracts/new", response_class=HTMLResponse)
async def new_contract_form(request: Request):
    """Show the form to create a new contract."""
    return templates.TemplateResponse("contract_form.html", {"request": request, "contract": None})


@app.post("/contracts/new")
async def create_contract_route(request: Request):
    """Handle submission of a new contract."""
    # Parse application/x-www-form-urlencoded body manually because python-multipart is not available
    body = await request.body()
    data = urllib.parse.parse_qs(body.decode('utf-8')) if body else {}
    name = (data.get('name', [''])[0] or '').strip()
    start_date = data.get('start_date', [''])[0]
    end_date = data.get('end_date', [''])[0]
    value_str = data.get('value', [''])[0]
    description = data.get('description', [''])[0]
    # Validate numeric value
    try:
        value = float(value_str)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Vrednost mora biti število.")
    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d").date()
        ed = datetime.strptime(end_date, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Neveljaven datum.")
    if ed < sd:
        raise HTTPException(status_code=400, detail="Datum konca mora biti po datumu začetka.")
    insert_contract(name, sd, ed, value, description)
    return RedirectResponse(url="/contracts", status_code=303)


@app.get("/contracts/edit/{contract_id}", response_class=HTMLResponse)
async def edit_contract_form(request: Request, contract_id: int):
    """Show the form to edit an existing contract."""
    contract = get_contract_by_id(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Pogodba ni bila najdena.")
    return templates.TemplateResponse(
        "contract_form.html",
        {
            "request": request,
            "contract": contract,
        },
    )


@app.post("/contracts/edit/{contract_id}")
async def update_contract_route(request: Request, contract_id: int):
    """Handle updates to an existing contract."""
    body = await request.body()
    data = urllib.parse.parse_qs(body.decode('utf-8')) if body else {}
    name = (data.get('name', [''])[0] or '').strip()
    start_date = data.get('start_date', [''])[0]
    end_date = data.get('end_date', [''])[0]
    value_str = data.get('value', [''])[0]
    description = data.get('description', [''])[0]
    try:
        value = float(value_str)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Vrednost mora biti število.")
    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d").date()
        ed = datetime.strptime(end_date, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Neveljaven datum.")
    if ed < sd:
        raise HTTPException(status_code=400, detail="Datum konca mora biti po datumu začetka.")
    update_contract(contract_id, name, sd, ed, value, description)
    return RedirectResponse(url="/contracts", status_code=303)


@app.post("/contracts/delete/{contract_id}")
async def delete_contract_route(contract_id: int):
    """Delete a contract by ID and redirect back to the contracts list."""
    delete_contract(contract_id)
    return RedirectResponse(url="/contracts", status_code=303)


@app.get("/urgency/{level}", response_class=HTMLResponse)
async def urgency_view(request: Request, level: str):
    """Show contracts filtered by urgency level (kritično, opozorilo, informativno, poteklo)."""
    valid_levels = {"kritično", "opozorilo", "informativno", "poteklo"}
    if level not in valid_levels:
        raise HTTPException(status_code=404, detail="Napačna stopnja nujnosti.")
    contracts = fetch_contracts(urgency_filter=level)
    return templates.TemplateResponse(
        "urgency.html",
        {
            "request": request,
            "contracts": contracts,
            "level": level,
        },
    )


@app.get("/import", response_class=HTMLResponse)
async def import_form(request: Request):
    """Display the import page with drag-and-drop upload and template download."""
    return templates.TemplateResponse(
        "import.html",
        {
            "request": request,
            "errors": [],
        },
    )


@app.post("/import/preview", response_class=HTMLResponse)
async def import_preview(request: Request):
    """Handle uploading a file via JSON (base64 encoded), parse it, validate columns, and render preview."""
    # Attempt to parse JSON body containing 'filename' and 'content'
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Neveljavna zahteva (pričakovan JSON).")
    filename = (data.get("filename") or "").strip()
    content_b64 = data.get("content")
    if not filename or content_b64 is None:
        return templates.TemplateResponse(
            "import.html",
            {
                "request": request,
                "errors": ["Manjkajo ime datoteke ali vsebina."],
            },
        )
    if not allowed_file(filename):
        return templates.TemplateResponse(
            "import.html",
            {
                "request": request,
                "errors": ["Dovoljene so samo datoteke vrste CSV in Excel (.xls, .xlsx)."],
            },
        )
    try:
        content_bytes = base64.b64decode(content_b64.encode('utf-8'))
    except Exception:
        return templates.TemplateResponse(
            "import.html",
            {
                "request": request,
                "errors": ["Vsebina datoteke ni pravilno kodirana."],
            },
        )
    # Read into pandas DataFrame
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content_bytes))
        else:
            df = pd.read_excel(io.BytesIO(content_bytes))
    except Exception as e:
        return templates.TemplateResponse(
            "import.html",
            {
                "request": request,
                "errors": [f"Napaka pri branju datoteke: {e}"],
            },
        )
    required_cols = {"name", "start_date", "end_date", "value"}
    df_cols = set(col.lower() for col in df.columns)
    missing = required_cols - df_cols
    if missing:
        return templates.TemplateResponse(
            "import.html",
            {
                "request": request,
                "errors": [
                    f"Manjkajo obvezni stolpci: {', '.join(sorted(missing))}. Uporabite predlogo za pravilen format."
                ],
            },
        )
    # Normalize columns to lowercase
    df = df.rename(columns={c: c.lower() for c in df.columns})
    preview_records: List[Dict] = []
    row_errors: List[str] = []
    for idx, row in df.iterrows():
        record = {
            "name": str(row.get("name", "")).strip(),
            "start_date": row.get("start_date"),
            "end_date": row.get("end_date"),
            "value": row.get("value"),
            "description": str(row.get("description", "")) if "description" in df.columns else "",
            "row_index": idx + 1,
        }
        # Validate numeric value
        try:
            record["value"] = float(record["value"])
        except Exception:
            row_errors.append(f"Vrstica {idx + 1}: vrednost '{record['value']}' ni število.")
        # Validate dates
        try:
            if isinstance(record["start_date"], pd.Timestamp):
                record["start_date"] = record["start_date"].date()
            else:
                record["start_date"] = datetime.strptime(str(record["start_date"]), "%Y-%m-%d").date()
        except Exception:
            row_errors.append(
                f"Vrstica {idx + 1}: začetni datum '{record['start_date']}' ni v formatu YYYY-MM-DD."
            )
        try:
            if isinstance(record["end_date"], pd.Timestamp):
                record["end_date"] = record["end_date"].date()
            else:
                record["end_date"] = datetime.strptime(str(record["end_date"]), "%Y-%m-%d").date()
        except Exception:
            row_errors.append(
                f"Vrstica {idx + 1}: končni datum '{record['end_date']}' ni v formatu YYYY-MM-DD."
            )
        if (
            isinstance(record.get("start_date"), date)
            and isinstance(record.get("end_date"), date)
            and record["end_date"] < record["start_date"]
        ):
            row_errors.append(
                f"Vrstica {idx + 1}: končni datum mora biti po začetnem datumu."
            )
        preview_records.append(record)
    token = str(uuid.uuid4())
    preview_store[token] = preview_records
    return templates.TemplateResponse(
        "import_preview.html",
        {
            "request": request,
            "records": preview_records,
            "errors": row_errors,
            "token": token,
        },
    )


@app.post("/import/confirm", response_class=HTMLResponse)
async def import_confirm(request: Request):
    """Finalize import of previewed records into the database."""
    # Retrieve token from body (application/x-www-form-urlencoded)
    body = await request.body()
    data = urllib.parse.parse_qs(body.decode('utf-8')) if body else {}
    token = data.get('token', [None])[0]
    if not token:
        raise HTTPException(status_code=400, detail="Manjka potrditveni token.")
    records = preview_store.pop(token, None)
    if records is None:
        raise HTTPException(status_code=400, detail="Potrditveni token ni veljaven ali je bil že uporabljen.")
    # Insert each record into DB
    for rec in records:
        # Skip records with invalid date or value
        if not isinstance(rec.get("start_date"), date) or not isinstance(rec.get("end_date"), date):
            continue
        try:
            val = float(rec.get("value"))
        except Exception:
            continue
        insert_contract(rec["name"], rec["start_date"], rec["end_date"], val, rec.get("description", ""))
    return templates.TemplateResponse(
        "import_success.html",
        {"request": request, "count": len(records)},
    )


@app.get("/template")
async def download_template():
    """Serve a simple Excel template for contract import."""
    # Create an in-memory template using pandas and return as attachment
    df = pd.DataFrame(
        [
            {
                "name": "Primer pogodbe",
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "value": 10000.0,
                "description": "Opis opcijski",
            }
        ]
    )
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    headers = {"Content-Disposition": "attachment; filename=predloga_pogodb.xlsx"}
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


# If run directly, start uvicorn (helpful for local dev/testing). This will not run automatically in the evaluation environment, but is kept for completeness.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)