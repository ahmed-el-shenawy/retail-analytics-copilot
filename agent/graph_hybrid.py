from typing import TypedDict, Literal, Any
from langgraph.graph import StateGraph, END
import json
import re

from agent.dspy_signatures import QuestionRouter, NL2SQLModule, SynthesizerModule
from agent.rag.retrieval import DocumentRetriever
from agent.tools.sqlite_tool import get_schema_text, execute_sql, extract_tables_from_sql


# ------------------------------
# State Definition
# ------------------------------
class AgentState(TypedDict):
    question: str
    format_hint: str
    route: str
    retrieved_chunks: list[dict]
    constraints: dict
    sql_query: str
    sql_results: dict
    sql_error: str | None
    final_answer: Any
    explanation: str
    confidence: float
    citations: list[str]
    repair_count: int
    max_repairs: int


# ------------------------------
# Hybrid Agent
# ------------------------------
class HybridAgent:
    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.retriever = DocumentRetriever()
        self.router = QuestionRouter()
        self.nl2sql = NL2SQLModule()
        self.synthesizer = SynthesizerModule()
        self.schema = get_schema_text()  # ← FIX: Use text format

    def log(self, *args):
        if self.enable_logging:
            print(*args)

    # ------------------------------
    # Nodes
    # ------------------------------
    def router_node(self, state: AgentState) -> AgentState:
        self.log("📍 Router: Classifying question...")
        route = self.router(state['question'])
        self.log(f"   → Route: {route}")
        return {**state, 'route': route}

    def retriever_node(self, state: AgentState) -> AgentState:
        if state['route'] in ['rag', 'hybrid']:
            self.log("📍 Retriever: Searching documents...")
            chunks = self.retriever.search(state['question'], top_k=3)
            self.log(f"   → Retrieved {len(chunks)} chunks")
            for chunk in chunks:
                self.log(f"      - {chunk['id']} (score: {chunk['score']:.2f})")
            return {**state, 'retrieved_chunks': chunks}
        return state

    def planner_node(self, state: AgentState) -> AgentState:
        if state['route'] in ['sql', 'hybrid']:
            self.log("📍 Planner: Extracting constraints...")
            constraints = self._extract_constraints(
                state['question'], state.get('retrieved_chunks', [])
            )
            self.log(f"   → Constraints: {json.dumps(constraints, indent=4)}")
            return {**state, 'constraints': constraints}
        return state

    def nl2sql_node(self, state: AgentState) -> AgentState:
        if state['route'] in ['sql', 'hybrid']:
            self.log("📍 NL2SQL: Generating SQL query...")
            error_feedback = state.get('sql_error') if state.get('repair_count', 0) > 0 else None
            if error_feedback:
                self.log(f"   ⚠️  Repair attempt {state['repair_count']}, previous error: {error_feedback}")
            
            result = self.nl2sql(
                state['question'],
                self.schema,
                state.get('constraints', {}),
                error_feedback=error_feedback
            )
            sql = re.sub(r'^```sql\n|```$', '', result.sql.strip(), flags=re.MULTILINE)
            self.log(f"   → Generated SQL:\n      {sql}")
            return {**state, 'sql_query': sql}
        return state

    def executor_node(self, state: AgentState) -> AgentState:
        if state.get('sql_query'):
            self.log("📍 Executor: Running SQL...")
            result = execute_sql(state['sql_query'])
            if result['success']:
                self.log(f"   ✓ Success: {len(result['rows'])} rows returned")
                if result['rows']:
                    self.log(f"   Sample: {result['rows'][0]}")
            else:
                self.log(f"   ✗ Error: {result['error']}")
            return {**state, 'sql_results': result, 'sql_error': result.get('error')}
        return state

    def repair_node(self, state: AgentState) -> AgentState:
        self.log("📍 Repair: Incrementing repair count for SQL")
        return {**state, 'repair_count': state.get('repair_count', 0) + 1}

    def synthesizer_node(self, state: AgentState) -> AgentState:
        self.log("📍 Synthesizer: Creating final answer...")
        result = self.synthesizer(
            state['question'],
            state.get('retrieved_chunks', []),
            state.get('sql_results', {}),
            state['format_hint']
        )
        final_answer = self._parse_answer(result.answer, state['format_hint'], state.get('sql_results', {}))
        citations = self._collect_citations(state)
        confidence = self._calculate_confidence(state, result)

        self.log(f"   → Answer: {final_answer}")
        self.log(f"   → Confidence: {confidence:.2f}")
        self.log(f"   → Citations: {citations}")

        return {
            **state,
            'final_answer': final_answer,
            'explanation': result.explanation,
            'confidence': confidence,
            'citations': citations
        }

    # ------------------------------
    # Helper Methods
    # ------------------------------
    def _extract_constraints(self, question, chunks):
        """
        Extract constraints from retrieved docs and question.
        CRITICAL FIX: Don't over-constrain category when question asks "which category"
        """
        constraints = {}
        date_pattern = r'(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'
        categories = ['Beverages', 'Condiments', 'Confections', 
                      'Dairy Products', 'Grains/Cereals', 'Meat/Poultry',
                      'Produce', 'Seafood']

        # Check if question is ASKING ABOUT categories (not specifying one)
        asking_about_category = any(phrase in question.lower() for phrase in [
            'which category', 'which product category', 'what category',
            'top category', 'highest category', 'best category'
        ])

        for chunk in chunks:
            content = chunk['content']

            # Extract date ranges
            date_matches = re.findall(date_pattern, content)
            if date_matches: 
                constraints['date_range'] = date_matches[0]

            # ✅ FIX: Only constrain category if NOT asking "which category"
            if not asking_about_category:
                for cat in categories:
                    # Only from retrieved docs, not from question
                    if cat in content:
                        constraints['category'] = cat
                        break

            # Extract KPI info
            if 'AOV' in question or 'Average Order Value' in question:
                if 'AOV' in content:
                    constraints['kpi_type'] = 'AOV'
                    constraints['kpi_formula'] = 'SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)'

            if 'gross margin' in question.lower() or 'margin' in question.lower():
                if 'Gross Margin' in content or 'margin' in content.lower():
                    constraints['kpi_type'] = 'gross_margin'
                    constraints['cost_approximation'] = 0.7

        return constraints

    def _parse_answer(self, answer_str, format_hint, sql_results=None):
        """
        Parse answer with fallback to SQL results if synthesis fails
        """
        try:
            answer_str = str(answer_str).strip()
            answer_str = re.sub(r"^```(json|sql)?\n|```$", "", answer_str, flags=re.MULTILINE)
            
            if format_hint == 'int':
                match = re.search(r'\d+', answer_str)
                return int(match.group()) if match else 0
            
            elif format_hint == 'float':
                match = re.search(r'[\d.]+', answer_str)
                if match:
                    return round(float(match.group()), 2)
                # Fallback: try to get from SQL results
                if sql_results and sql_results.get('rows'):
                    first_row = sql_results['rows'][0]
                    for value in first_row.values():
                        if isinstance(value, (int, float)):
                            return round(float(value), 2)
                return 0.0
            
            elif format_hint.startswith('{'):
                # Try to parse JSON object
                try:
                    return json.loads(answer_str)
                except:
                    # Fallback: construct from SQL results
                    if sql_results and sql_results.get('rows') and sql_results['rows']:
                        return sql_results['rows'][0]
                    return {}
            
            elif format_hint.startswith('list'):
                try:
                    return json.loads(answer_str)
                except:
                    # Fallback: return SQL rows directly
                    if sql_results and sql_results.get('rows'):
                        return sql_results['rows']
                    return []
            
            return answer_str
            
        except Exception as e:
            self.log(f"   ⚠️  Parse error: {e}, returning fallback")
            # Try to use SQL results directly
            if sql_results and sql_results.get('rows'):
                if format_hint.startswith('list'):
                    return sql_results['rows']
                elif format_hint.startswith('{') and sql_results['rows']:
                    return sql_results['rows'][0]
            return answer_str

    def _collect_citations(self, state):
        citations = []
        if state.get('sql_query'):
            citations.extend(extract_tables_from_sql(state['sql_query']))
        if state.get('retrieved_chunks'):
            citations.extend([c['id'] for c in state['retrieved_chunks']])
        return list(set(citations))

    def _calculate_confidence(self, state, synth_result):
        confidence = 0.5
        sql_res = state.get("sql_results", {})
        
        # SQL success bonus
        if sql_res.get("success"): 
            confidence += 0.2
            # Extra bonus for non-empty results
            if sql_res.get("rows") and len(sql_res["rows"]) > 0:
                confidence += 0.1
        
        # Retrieval quality
        if state.get("retrieved_chunks"):
            avg_score = sum(c.get("score", 0) for c in state["retrieved_chunks"]) / max(1, len(state["retrieved_chunks"]))
            confidence += avg_score * 0.2
        
        # Repair penalty
        confidence += 0.1 if state.get("repair_count", 0) == 0 else -0.05 * state["repair_count"]
        
        # DSPy confidence
        try: 
            confidence = (confidence + float(synth_result.confidence)) / 2
        except: 
            pass
        
        return min(max(confidence, 0.0), 1.0)

    # ------------------------------
    # Conditional Edge Functions
    # ------------------------------
    def should_repair(self, state: AgentState) -> Literal['repair', 'synthesize']:
        if state.get('sql_error') and state.get('repair_count', 0) < state.get('max_repairs', 2):
            return 'repair'
        return 'synthesize'

    def route_after_router(self, state: AgentState) -> Literal['retriever', 'planner']:
        return 'retriever' if state['route'] in ['rag', 'hybrid'] else 'planner'

    # ------------------------------
    # Graph Construction
    # ------------------------------
    def build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self.router_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("nl2sql", self.nl2sql_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("repair", self.repair_node)
        workflow.add_node("synthesizer", self.synthesizer_node)

        workflow.set_entry_point("router")
        workflow.add_conditional_edges("router", self.route_after_router, {'retriever': 'retriever', 'planner': 'planner'})
        workflow.add_edge("retriever", "planner")
        workflow.add_edge("planner", "nl2sql")
        workflow.add_edge("nl2sql", "executor")
        workflow.add_conditional_edges("executor", self.should_repair, {'repair': 'repair', 'synthesize': 'synthesizer'})
        workflow.add_edge("repair", "nl2sql")
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def run(self, question: str, format_hint: str, max_repairs: int = 2):
        graph = self.build_graph()
        initial_state: AgentState = {
            'question': question,
            'format_hint': format_hint,
            'route': '',
            'retrieved_chunks': [],
            'constraints': {},
            'sql_query': '',
            'sql_results': {},
            'sql_error': None,
            'final_answer': None,
            'explanation': '',
            'confidence': 0.0,
            'citations': [],
            'repair_count': 0,
            'max_repairs': max_repairs
        }
        return graph.invoke(initial_state)