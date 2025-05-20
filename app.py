import gradio as gr
import torch
import joblib
from sentence_transformers import SentenceTransformer, util

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings = joblib.load('doc_embeddings.pt')
questions = joblib.load('questions.pkl')
answers = joblib.load('answers.pkl')

# Define retrieval function
def retrieve_documents(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)
    top_results = torch.topk(cos_scores, k=top_k)

    results = ""
    for score, idx in zip(top_results[0][0], top_results[1][0]):
        results += f"🔍 **Q**: {questions[idx]}\n\n🩺 **A**: {answers[idx]}\n\n📈 **Score**: {score.item():.4f}\n\n---\n\n"
    return results

# Build the Gradio interface
iface = gr.Interface(
    fn=retrieve_documents,
    inputs=gr.Textbox(lines=2, placeholder="Enter your medical question here...", label="Medical Query"),
    outputs=gr.Markdown(label="Top Relevant Answers"),
    title="Dense Retriever (SBERT) - Medical Q&A",
    description="Ask a medical question. The retriever will return the most relevant answers from MedQuAD dataset using SBERT embeddings.",
    examples=[
        ["What are the symptoms of asthma?"],
        ["How is diabetes treated?"],
        ["What causes migraine headaches?"]
    ]
)

# Launch the app
iface.launch()
