import gradio as gr
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Load a transformer model for math reasoning
model_name = "google/flan-t5-large"  # Can be fine-tuned later
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def solve_math_problem(problem):
    """Generates a solution for the given math problem using the trained AI model."""
    solution = pipe(problem, max_length=512, truncation=True)
    return solution[0]['generated_text']

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## AI Agent for Solving the 7 Hardest Math Problems")
    math_input = gr.Textbox(label="Enter a Math Problem")
    submit_button = gr.Button("Solve")
    output = gr.Textbox(label="AI Solution")

    submit_button.click(fn=solve_math_problem, inputs=math_input, outputs=output)

demo.launch()
