import sqlite3
import re
from typing import List, Dict, Any, Optional
from functools import lru_cache

DB_PATH = "data/northwind.sqlite"

# ==============================================================================
# SCHEMA RETRIEVAL - Critical Fix: Include ALL tables with proper names
# ==============================================================================

@lru_cache(maxsize=1)
def get_schema_info() -> Dict[str, List[Dict[str, str]]]:
    """
    Returns complete database schema as a structured dictionary.
    
    Returns:
        Dict mapping table names to their column definitions.
        Example: {
            'Orders': [
                {'name': 'OrderID', 'type': 'INTEGER'},
                {'name': 'CustomerID', 'type': 'VARCHAR'},
                ...
            ]
        }
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Get ALL tables (not just views)
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            # Handle tables with special characters (like "Order Details")
            table_query = f'"{table}"' if ' ' in table else table
            cursor.execute(f"PRAGMA table_info({table_query})")
            columns = cursor.fetchall()
            
            schema[table] = [
                {
                    'name': col[1],
                    'type': col[2],
                    'notnull': col[3],
                    'pk': col[5]
                }
                for col in columns
            ]
        
        return schema
        
    finally:
        conn.close()


def get_schema_text() -> str:
    """
    Returns schema formatted as human-readable text for LLM prompts.
    This is what should be passed to NL2SQL module.
    """
    schema = get_schema_info()
    
    lines = ["=== NORTHWIND DATABASE SCHEMA ===\n"]
    
    for table_name, columns in schema.items():
        # Quote table names with spaces
        display_name = f'"{table_name}"' if ' ' in table_name else table_name
        lines.append(f"Table: {display_name}")
        
        for col in columns:
            pk_marker = " [PRIMARY KEY]" if col['pk'] else ""
            notnull_marker = " [NOT NULL]" if col['notnull'] else ""
            lines.append(f"  - {col['name']} ({col['type']}){pk_marker}{notnull_marker}")
        
        lines.append("")  # Blank line between tables
    
    # Add critical notes
    lines.append("=== IMPORTANT NOTES ===")
    lines.append("• Table 'Order Details' MUST be quoted: \"Order Details\"")
    lines.append("• Use lowercase view aliases: orders, products, customers")
    lines.append("• Revenue calculation: SUM(UnitPrice * Quantity * (1 - Discount))")
    lines.append("• Date filtering: WHERE DATE(OrderDate) BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'")
    lines.append("• Join Products to Categories using: Products.CategoryID = Categories.CategoryID")
    
    return "\n".join(lines)


# ==============================================================================
# SQL EXECUTION - Enhanced error handling and logging
# ==============================================================================

def execute_sql(query: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Execute SQL query with robust error handling.
    
    Args:
        query: SQL query string
        verbose: If True, print detailed execution info
        
    Returns:
        Dict with keys:
        - success: bool
        - rows: list of dicts (empty on failure)
        - columns: list of column names
        - error: error message (None on success)
        - row_count: number of rows returned
    """
    if not query or not isinstance(query, str):
        return {
            "success": False,
            "error": "Invalid query: empty or not a string",
            "rows": [],
            "columns": [],
            "row_count": 0
        }
    
    # Clean the query
    query = query.strip()
    if not query:
        return {
            "success": False,
            "error": "Query is empty after cleaning",
            "rows": [],
            "columns": [],
            "row_count": 0
        }
    
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        cursor = conn.cursor()
        
        if verbose:
            print(f"   [SQL] Executing query:")
            print(f"   {query[:200]}..." if len(query) > 200 else f"   {query}")
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        result_rows = [dict(row) for row in rows]
        columns = list(rows[0].keys()) if rows else []
        
        if verbose:
            print(f"   [SQL] Success: {len(result_rows)} rows returned")
            if result_rows:
                print(f"   [SQL] Sample row: {result_rows[0]}")
        
        return {
            "success": True,
            "rows": result_rows,
            "columns": columns,
            "error": None,
            "row_count": len(result_rows)
        }
        
    except sqlite3.Error as e:
        error_msg = str(e)
        
        if verbose:
            print(f"   [SQL] Error: {error_msg}")
        
        # Enhanced error messages with hints
        hints = _get_error_hints(error_msg, query)
        full_error = f"{error_msg}{hints}"
        
        return {
            "success": False,
            "error": full_error,
            "rows": [],
            "columns": [],
            "row_count": 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "rows": [],
            "columns": [],
            "row_count": 0
        }
        
    finally:
        if conn:
            conn.close()


def _get_error_hints(error_msg: str, query: str) -> str:
    """
    Provide helpful hints based on common SQL errors.
    """
    hints = []
    
    # No such table
    if "no such table" in error_msg.lower():
        # Extract table name from error
        match = re.search(r"no such table:\s*(\S+)", error_msg, re.IGNORECASE)
        if match:
            wrong_table = match.group(1)
            hints.append(f"\n  → Table '{wrong_table}' does not exist")
            
            # Common mistakes
            if wrong_table.lower() in ['orderdetails', 'order_details']:
                hints.append(f'  → Use "Order Details" (with quotes and space)')
            elif wrong_table.lower() == 'categories':
                hints.append(f"  → Use 'Categories' (capital C)")
            elif wrong_table.lower() in ['salestable', 'sales']:
                hints.append(f"  → No sales table exists. Use Orders + \"Order Details\"")
    
    # No such column
    elif "no such column" in error_msg.lower():
        match = re.search(r"no such column:\s*(\S+)", error_msg, re.IGNORECASE)
        if match:
            wrong_col = match.group(1)
            hints.append(f"\n  → Column '{wrong_col}' does not exist")
            
            # Common mistakes
            if 'categoryname' in wrong_col.lower():
                hints.append(f"  → CategoryName is in 'Categories' table, not 'Products'")
                hints.append(f"  → Join: Products.CategoryID = Categories.CategoryID")
            elif 'returnwindow' in wrong_col.lower():
                hints.append(f"  → No return policy data in database")
                hints.append(f"  → This info is in DOCUMENTS (use RAG, not SQL)")
            elif 'productname' in wrong_col.lower() and 'order' in query.lower():
                hints.append(f"  → ProductName is in 'Products' table")
                hints.append(f"  → Join: \"Order Details\".ProductID = Products.ProductID")
    
    # Syntax errors
    elif "syntax error" in error_msg.lower():
        if "BETWEDIR" in query or "BETWELOG" in query:
            hints.append("\n  → Typo in BETWEEN clause")
            hints.append("  → Correct: WHERE DATE(col) BETWEEN 'date1' AND 'date2'")
    
    # Ambiguous column
    elif "ambiguous" in error_msg.lower():
        hints.append("\n  → Column name exists in multiple tables")
        hints.append("  → Use table prefix: table_name.column_name")
    
    return "".join(hints) if hints else ""


# ==============================================================================
# TABLE EXTRACTION - Improved regex
# ==============================================================================

def extract_tables_from_sql(sql: str) -> List[str]:
    """
    Extract all table names referenced in SQL query.
    Handles quoted tables, aliases, and various SQL clauses.
    
    Args:
        sql: SQL query string
        
    Returns:
        List of unique table names (case-preserved)
    """
    if not sql or not isinstance(sql, str):
        return []
    
    tables = set()
    
    # Pattern 1: FROM/JOIN with optional quotes
    # Matches: FROM "Order Details", JOIN products, etc.
    pattern1 = re.compile(
        r'\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+'
        r'(?:'
        r'"([^"]+)"|'           # Quoted with double quotes
        r"'([^']+)'|"           # Quoted with single quotes  
        r'`([^`]+)`|'           # Quoted with backticks
        r'(\w+)'                # Unquoted word
        r')',
        re.IGNORECASE
    )
    
    for match in pattern1.finditer(sql):
        table = match.group(1) or match.group(2) or match.group(3) or match.group(4)
        if table:
            # Remove any trailing aliases (e.g., "Orders o" -> "Orders")
            table = table.split()[0].strip().strip(',;')
            if table and not _is_sql_keyword(table):
                tables.add(table)
    
    return sorted(list(tables))


def _is_sql_keyword(word: str) -> bool:
    """Check if word is a SQL keyword (not a table name)"""
    keywords = {
        'SELECT', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT',
        'OFFSET', 'UNION', 'INTERSECT', 'EXCEPT', 'CASE', 'WHEN',
        'THEN', 'ELSE', 'END', 'AS', 'ON', 'USING', 'AND', 'OR',
        'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS', 'NULL',
        'DISTINCT', 'ALL', 'ASC', 'DESC'
    }
    return word.upper() in keywords


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def verify_database_setup() -> Dict[str, Any]:
    """
    Verify database is properly set up with expected tables.
    Useful for debugging setup issues.
    
    Returns:
        Dict with status and diagnostics
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check for key tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        # Expected core tables
        expected = ['Orders', 'Order Details', 'Products', 'Customers', 'Categories']
        missing = [t for t in expected if t not in tables]
        
        # Check if views exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='view'
        """)
        views = [row[0] for row in cursor.fetchall()]
        
        # Sample row count
        cursor.execute("SELECT COUNT(*) FROM Orders")
        order_count = cursor.fetchone()[0]
        
        status = {
            "success": len(missing) == 0,
            "tables_found": tables,
            "tables_missing": missing,
            "views_found": views,
            "sample_order_count": order_count,
            "database_path": DB_PATH
        }
        
        conn.close()
        return status
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "database_path": DB_PATH
        }


def get_table_sample(table_name: str, limit: int = 3) -> Dict[str, Any]:
    """
    Get sample rows from a table for debugging.
    
    Args:
        table_name: Name of table (will be quoted if needed)
        limit: Number of rows to return
        
    Returns:
        Dict with sample data or error
    """
    # Quote table name if it contains spaces
    quoted_table = f'"{table_name}"' if ' ' in table_name else table_name
    query = f"SELECT * FROM {quoted_table} LIMIT {limit}"
    
    return execute_sql(query)


# ==============================================================================
# SCHEMA VALIDATION FOR COMMON QUERIES
# ==============================================================================

def validate_query_against_schema(query: str) -> Dict[str, Any]:
    """
    Pre-validate a query before execution to catch common errors.
    
    Returns:
        Dict with 'valid' bool and 'warnings' list
    """
    warnings = []
    schema = get_schema_info()
    
    # Extract referenced tables
    tables = extract_tables_from_sql(query)
    
    # Check if tables exist
    for table in tables:
        if table not in schema and table.lower() not in [t.lower() for t in schema.keys()]:
            warnings.append(f"Table '{table}' not found in schema")
    
    # Check for common mistakes
    if 'Order Details' in query and '"Order Details"' not in query:
        warnings.append("Table 'Order Details' should be quoted")
    
    if re.search(r'\bCategoryName\b', query) and 'Categories' not in tables:
        warnings.append("CategoryName requires JOIN with Categories table")
    
    if re.search(r'\bProductName\b', query) and 'Products' not in tables:
        warnings.append("ProductName requires JOIN with Products table")
    
    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "tables_referenced": tables
    }


# ==============================================================================
# TESTING / DEBUG
# ==============================================================================

if __name__ == "__main__":
    """Quick test of database connection and schema retrieval"""
    
    print("=" * 80)
    print("DATABASE DIAGNOSTICS")
    print("=" * 80)
    
    # Test 1: Verify setup
    print("\n1. Verifying database setup...")
    status = verify_database_setup()
    print(f"   Status: {'✓ OK' if status['success'] else '✗ FAIL'}")
    print(f"   Tables: {', '.join(status.get('tables_found', []))}")
    if status.get('tables_missing'):
        print(f"   Missing: {', '.join(status['tables_missing'])}")
    print(f"   Order count: {status.get('sample_order_count', 0)}")
    
    # Test 2: Schema retrieval
    print("\n2. Testing schema retrieval...")
    schema = get_schema_info()
    print(f"   Found {len(schema)} tables")
    for table in list(schema.keys())[:5]:
        print(f"   - {table} ({len(schema[table])} columns)")
    
    # Test 3: Sample query
    print("\n3. Testing sample query...")
    result = execute_sql("SELECT COUNT(*) as total FROM Orders", verbose=True)
    if result['success']:
        print(f"   ✓ Query successful: {result['rows']}")
    else:
        print(f"   ✗ Query failed: {result['error']}")
    
    # Test 4: Schema text format
    print("\n4. Schema text format (first 500 chars)...")
    schema_text = get_schema_text()
    print(schema_text[:500] + "...")
    
    print("\n" + "=" * 80)