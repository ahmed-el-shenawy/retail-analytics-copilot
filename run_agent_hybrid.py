#!/usr/bin/env python3
import json
import click
from rich.console import Console
from rich.progress import track
import dspy

from agent.graph_hybrid import HybridAgent

console = Console()

def setup_dspy():
    """Configure DSPy with local Ollama model"""
    try:
        # DSPy 2.4+ recommended format
        lm = dspy.LM(
            model='ollama/qwen3:4b-instruct',
            api_base='http://localhost:11434',
            api_key='',  # Not needed for Ollama but required param
        )
        dspy.configure(lm=lm)
        
        console.print("   ✓ DSPy configured successfully")
        
    except Exception as e:
        console.print(f"[bold red]✗ DSPy configuration failed: {e}[/bold red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("1. Ensure Ollama is running: ollama serve")
        console.print("2. Check model is pulled: ollama list")
        console.print("3. Pull if needed: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M")
        raise

@click.command()
@click.option('--batch', required=True, help='Input JSONL file with questions')
@click.option('--out', required=True, help='Output JSONL file for results')
def main(batch, out):
    """
    Run Retail Analytics Copilot in batch mode
    
    Example:
        python run_agent_hybrid.py \\
            --batch sample_questions_hybrid_eval.jsonl \\
            --out outputs_hybrid.jsonl
    """
    console.print("[bold blue]🚀 Retail Analytics Copilot[/bold blue]")
    console.print(f"📥 Input: {batch}")
    console.print(f"📤 Output: {out}\n")
    
    # Setup DSPy
    console.print("⚙️  Configuring DSPy with Ollama...")
    setup_dspy()
    
    # Initialize agent
    console.print("🤖 Initializing agent...\n")
    agent = HybridAgent()
    
    # Load questions
    with open(batch, 'r') as f:
        questions = [json.loads(line) for line in f]
    
    console.print(f"📋 Processing {len(questions)} questions...\n")
    
    # Process each question
    results = []
    for q in track(questions, description="Running agent..."):
        console.print(f"\n{'='*80}")
        console.print(f"[bold]Question ID:[/bold] {q['id']}")
        console.print(f"[bold]Question:[/bold] {q['question']}")
        console.print(f"[bold]Format:[/bold] {q['format_hint']}\n")
        
        try:
            # Run agent
            result = agent.run(
                question=q['question'],
                format_hint=q['format_hint'],
                max_repairs=2
            )
            
            # Format output
            output = {
                'id': q['id'],
                'final_answer': result['final_answer'],
                'sql': result.get('sql_query', ''),
                'confidence': result['confidence'],
                'explanation': result['explanation'],
                'citations': result['citations']
            }
            
            results.append(output)
            
            console.print(f"\n[bold green]✓ Success[/bold green]")
            console.print(f"Answer: {output['final_answer']}")
            console.print(f"Confidence: {output['confidence']:.2f}")
            
        except Exception as e:
            console.print(f"\n[bold red]✗ Error:[/bold red] {str(e)}")
            # Write error output
            output = {
                'id': q['id'],
                'final_answer': None,
                'sql': '',
                'confidence': 0.0,
                'explanation': f"Error: {str(e)}",
                'citations': []
            }
            results.append(output)
    
    # Write outputs
    console.print(f"\n{'='*80}")
    console.print(f"💾 Writing results to {out}...")
    with open(out, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    console.print("[bold green]✨ Done![/bold green]")

if __name__ == '__main__':
    main()