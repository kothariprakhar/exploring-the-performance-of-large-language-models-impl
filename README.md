# Exploring the Performance of Large Language Models on Subjective Span Identification Tasks

Identifying relevant text spans is important for several downstream tasks in NLP, as it contributes to model explainability. While most span identification approaches rely on relatively smaller pre-trained language models like BERT, a few recent approaches have leveraged the latest generation of Large Language Models (LLMs) for the task. Current work has focused on explicit span identification like Named Entity Recognition (NER), while more subjective span identification with LLMs in tasks like Aspect-based Sentiment Analysis (ABSA) has been underexplored. In this paper, we fill this important gap by presenting an evaluation of the performance of various LLMs on text span identification in three popular tasks, namely sentiment analysis, offensive language identification, and claim verification. We explore several LLM strategies like instruction tuning, in-context learning, and chain of thought. Our results indicate underlying relationships within text aid LLMs in identifying precise text spans.

## Implementation Details

# Subjective Span Identification with LLMs

This implementation mimics the experimental framework described in the paper "Exploring the Performance of Large Language Models on Subjective Span Identification Tasks." 

## Core Concepts & Architecture

### 1. Subjective Span Identification
The paper differentiates between objective tasks (like Named Entity Recognition, where "Paris" is objectively a location) and **subjective tasks**. In the provided code, the `raw_data` dictionary simulates three subjective domains:
*   **Sentiment Analysis:** Identifying the specific phrase justifying a sentiment (e.g., "service was slow").
*   **Offensive Language:** Identifying the specific insult.
*   **Claim Verification:** Isolating the claim being made.

### 2. Model Architecture: Seq2Seq LLM
The paper evaluates Generative LLMs (like GPT or FLAN-T5) rather than discriminative encoders (like BERT). 
*   **Code Mapping:** We use `google/flan-t5-small` via the Hugging Face `AutoModelForSeq2SeqLM`. This model is an Encoder-Decoder architecture pre-trained on instructions, making it a suitable lightweight proxy for the larger LLMs discussed in the paper.
*   **Why Generative?** Unlike BERT, which classifies token indices (Start/End), the LLM generates the text span directly as a string sequence. This requires different evaluation metrics (overlap) rather than strict index matching.

## Implemented LLM Strategies

The paper explores how different prompting and training strategies affect performance. The code implements these via the `PromptStrategy` class:

### A. Instruction Tuning (Fine-tuning)
The abstract mentions "Instruction Tuning." In the code, **Experiment 2** implements a standard PyTorch training loop.
*   **Math:** We minimize the Cross-Entropy Loss between the generated tokens and the gold span tokens: $L = -\sum \log P(y_t | y_{<t}, x)$.
*   **Implementation:** The `SubjectiveSpanDataset` formats inputs as instructions (e.g., "Task: Identify the text span..."). The model weights are updated to maximize the likelihood of producing the `gold_span` given this instruction.

### B. In-Context Learning (Few-Shot)
The abstract mentions "In-Context Learning" (ICL). This allows the model to learn from examples in the prompt without updating weights.
*   **Code Mapping:** The `PromptStrategy.few_shot` method dynamically retrieves examples from the dataset and prepends them to the input. 
*   **Logic:** $Prompt = E_1 + E_2 + ... + E_n + Target_{input}$, where $E_i$ contains an input-label-span triplet. This guides the model's generation process by analogy.

### C. Chain of Thought (CoT)
The abstract mentions exploring "Chain of Thought." 
*   **Code Mapping:** The `PromptStrategy.chain_of_thought` method alters the prompt to explicitly ask the model to "think step by step." 
*   **Concept:** Instead of mapping $Input \rightarrow Span$, the model is prompted to map $Input \rightarrow Reasoning \rightarrow Span$. This is particularly useful for subjective tasks where the definition of "offensive" or "sentiment" might require contextual reasoning.

## Evaluation Metric
Since the model generates text, we cannot use simple Accuracy. The code implements **Token F1 Score** (in `calculate_f1`). This measures the overlap between the bag-of-tokens in the predicted string and the gold standard string, which is standard for generative QA and span extraction tasks.