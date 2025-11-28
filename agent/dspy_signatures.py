import json
import dspy

# ==============================================================================
# SIGNATURES
# ==============================================================================

class RouterSignature(dspy.Signature):
    """Classify whether question needs RAG docs, SQL query, or both"""
    question: str = dspy.InputField()
    route: str = dspy.OutputField(
        desc="Exactly one of: 'rag' (policy/docs only), 'sql' (database only), 'hybrid' (needs both docs AND database)"
    )


class NL2SQLSignature(dspy.Signature):
    """Generate valid SQLite query from natural language question"""
    question: str = dspy.InputField()
    db_schema: str = dspy.InputField(desc="Available database tables and columns")
    constraints: str = dspy.InputField(desc="Extracted date ranges, categories, KPIs")
    error_feedback: str = dspy.InputField(default="", desc="Previous error to fix")
    
    sql: str = dspy.OutputField(desc="ONLY the SQL query, no explanations")


class SynthesizerSignature(dspy.Signature):
    """Create final answer from retrieved documents and SQL results"""
    question: str = dspy.InputField()
    doc_chunks: str = dspy.InputField(desc="Retrieved document content")
    sql_results: str = dspy.InputField(desc="SQL query results")
    format_hint: str = dspy.InputField(desc="Required output format (int, float, dict, list)")
    
    answer: str = dspy.OutputField(desc="Final answer in requested format")
    explanation: str = dspy.OutputField(desc="Brief 1-2 sentence explanation")
    confidence: str = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")


# ==============================================================================
# MODULES
# ==============================================================================

class QuestionRouter(dspy.Module):
    """Classify question type with improved prompting"""
    
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouterSignature)
    
    def forward(self, question):
        # Hardcoded rules for reliability
        q_lower = question.lower()
        
        # Rule 1: Pure RAG questions
        if any(word in q_lower for word in ['policy', 'return window', 'return days', 'definition']):
            return 'rag'
        
        # Rule 2: Hybrid questions (needs both docs + SQL)
        if any(word in q_lower for word in ['during', 'summer', 'winter', 'calendar', 'marketing']):
            return 'hybrid'
        
        # Rule 3: Pure SQL (numbers, rankings, totals)
        if any(word in q_lower for word in ['top 3', 'total revenue', 'all-time', 'how many']):
            return 'sql'
        
        # Fallback to model classification
        try:
            enhanced_question = f"""Classify this question:
"{question}"

Guidelines:
- If asking about POLICIES, RETURN WINDOWS, DEFINITIONS from documents → route='rag'
- If asking for NUMBERS, TOTALS, RANKINGS from database → route='sql'  
- If needs BOTH document info (dates/categories) AND database numbers → route='hybrid'

Question: {question}"""
            
            result = self.classify(question=enhanced_question)
            route = result.route.strip().lower()
            
            if route not in ['rag', 'sql', 'hybrid']:
                route = 'hybrid'  # Safe default
            
            return route
            
        except Exception as e:
            print(f"   ⚠️  Router error: {e}, defaulting to 'hybrid'")
            return 'hybrid'


class NL2SQLModule(dspy.Module):
    """Generate SQL with strict schema enforcement"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(NL2SQLSignature)
    
    def forward(self, question, schema, constraints, error_feedback=None):
        # Format schema as readable text
        schema_text = self._format_schema(schema)
        
        # Format constraints as text
        constraints_text = json.dumps(constraints, indent=2) if constraints else "{}"
        
        # Enhanced error feedback
        if error_feedback:
            error_analysis = self._analyze_error(error_feedback)
            error_text = f"{error_feedback}\n\n{error_analysis}"
        else:
            error_text = "None"
        
        # Enhanced prompt for better SQL generation
        enhanced_question = f"""Generate ONLY valid SQLite SQL for this question.

Question: {question}

Available Tables and Columns:
{schema_text}

Extracted Constraints:
{constraints_text}

Previous Error (if any):
{error_text}

CRITICAL RULES:
1. Quote "Order Details": FROM "Order Details" od
2. Use EXACT table/column names from schema above
3. Date filter: WHERE o.OrderDate BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD' (no DATE() function!)
4. Revenue: SUM(od.UnitPrice * od.Quantity * (1 - od.Discount))
   ↑ Use Order Details.UnitPrice, NOT Products.UnitPrice!
5. ALWAYS include JOIN clause before using table alias:
   - JOIN Orders o ON od.OrderID = o.OrderID
   - JOIN Products p ON od.ProductID = p.ProductID
   - JOIN Categories c ON p.CategoryID = c.CategoryID
   - JOIN Customers cu ON o.CustomerID = cu.CustomerID
6. Return ONLY SQL, NO explanations

SQL Query:"""
        
        try:
            result = self.generate(
                question=enhanced_question,
                db_schema=schema_text,
                constraints=constraints_text,
                error_feedback=error_text
            )
            
            # Clean up the SQL
            sql = result.sql.strip()
            sql = sql.replace('```sql', '').replace('```', '')
            sql = sql.split('\n\n')[0]  # Take only first paragraph
            
            # Remove comments
            lines = [line for line in sql.split('\n') if not line.strip().startswith('--')]
            sql = '\n'.join(lines)
            
            return type('Result', (), {'sql': sql.strip(), 'reasoning': ''})()
            
        except Exception as e:
            print(f"   ⚠️  NL2SQL error: {e}")
            return type('Result', (), {'sql': 'SELECT 1;', 'reasoning': f'Error: {e}'})()
    
    def _analyze_error(self, error_msg):
        """Provide specific fix instructions based on error"""
        error_lower = error_msg.lower()
        
        if "no such column: o." in error_lower or "no such column: orders." in error_lower:
            return """
🔧 FIX REQUIRED: Missing Orders JOIN!

You referenced Orders columns but forgot to JOIN the table.

Add this line:
  JOIN Orders o ON od.OrderID = o.OrderID

Then you can use: o.OrderDate, o.CustomerID, etc.
"""
        
        elif "no such column: p." in error_lower or "no such column: products." in error_lower:
            return """
🔧 FIX REQUIRED: Missing Products JOIN!

Add this line:
  JOIN Products p ON od.ProductID = p.ProductID
"""
        
        elif "no such column: c." in error_lower or "no such column: categories." in error_lower:
            return """
🔧 FIX REQUIRED: Missing Categories JOIN!

Add this line:
  JOIN Categories c ON p.CategoryID = c.CategoryID
"""
        
        elif "no such column: cu." in error_lower or "no such column: customers." in error_lower:
            return """
🔧 FIX REQUIRED: Missing Customers JOIN!

Add this line:
  JOIN Customers cu ON o.CustomerID = cu.CustomerID
"""
        
        else:
            return "Review the error and fix the SQL accordingly."
    
    def _format_schema(self, schema):
        """Convert schema dict to readable text"""
        if isinstance(schema, str):
            return schema
        
        lines = ["DATABASE SCHEMA:"]
        for table, columns in schema.items():
            lines.append(f"\nTable: {table}")
            if isinstance(columns, list):
                for col in columns:
                    if isinstance(col, dict):
                        lines.append(f"  - {col.get('name', col)} ({col.get('type', 'TEXT')})")
                    else:
                        lines.append(f"  - {col}")
        
        return '\n'.join(lines)


class SynthesizerModule(dspy.Module):
    """Synthesize final answer with better formatting"""
    
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.Predict(SynthesizerSignature)
    
    def forward(self, question, doc_chunks, sql_results, format_hint):
        # Format inputs as text
        docs_text = self._format_docs(doc_chunks)
        sql_text = self._format_sql_results(sql_results)
        
        enhanced_question = f"""Synthesize the final answer for this question.

Question: {question}

Retrieved Documents:
{docs_text}

SQL Query Results:
{sql_text}

Required Format: {format_hint}

CRITICAL RULES:
1. If format_hint is 'int': return ONLY a single integer number
2. If format_hint is 'float': return ONLY a decimal number (e.g., 123.45)
3. If format_hint contains '{{': return valid JSON object (e.g., {{"category": "Beverages", "quantity": 100}})
4. If format_hint contains 'list': return valid JSON array (e.g., [{{"product": "X", "revenue": 100.0}}])
5. If SQL returned empty rows, check if you can infer the answer from context
6. NO extra text, NO markdown, JUST the answer in the requested format

Answer:"""
        
        try:
            result = self.synthesize(
                question=enhanced_question,
                doc_chunks=docs_text,
                sql_results=sql_text,
                format_hint=format_hint
            )
            
            return result
            
        except Exception as e:
            print(f"   ⚠️  Synthesizer error: {e}")
            return type('Result', (), {
                'answer': '0',
                'explanation': f'Error during synthesis: {e}',
                'confidence': '0.0'
            })()
    
    def _format_docs(self, doc_chunks):
        """Format document chunks as text"""
        if not doc_chunks:
            return "No documents retrieved"
        
        if isinstance(doc_chunks, str):
            return doc_chunks
        
        lines = []
        for chunk in doc_chunks:
            if isinstance(chunk, dict):
                lines.append(f"[{chunk.get('id', 'unknown')}]: {chunk.get('content', '')}")
            else:
                lines.append(str(chunk))
        
        return '\n'.join(lines) if lines else "No documents"
    
    def _format_sql_results(self, sql_results):
        """Format SQL results as text"""
        if not sql_results:
            return "No SQL executed"
        
        if isinstance(sql_results, str):
            return sql_results
        
        if isinstance(sql_results, dict):
            if sql_results.get('success'):
                rows = sql_results.get('rows', [])
                if rows:
                    return json.dumps(rows, indent=2)
                else:
                    return "SQL executed successfully but returned no rows"
            else:
                return f"SQL Error: {sql_results.get('error', 'Unknown error')}"
        
        return json.dumps(sql_results, indent=2)