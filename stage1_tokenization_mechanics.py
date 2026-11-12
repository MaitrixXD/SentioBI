"""
Stage 1: Tokenization, Logits, Softmax & Confidence Scores
===========================================================
Customer Feedback Intelligence System - Learning Lab

This script unpacks the 'black box' of Hugging Face Transformers.
Instead of relying only on pipeline(), we break down each internal step:
  Step 1: Text -> Tokens -> Input IDs + Attention Mask (Tokenizer)
  Step 2: Input IDs -> Hidden Representations -> Raw Logits (Model Forward Pass)
  Step 3: Logits -> Probabilities via Softmax (PyTorch)
  Step 4: Mapping Probabilities to Business Labels & Confidence Scores
  Step 5: Comparing manual calculation with Hugging Face's pipeline()
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def run_stage_1_lab():
    # Model: cardiffnlp/twitter-roberta-base-sentiment-latest
    # A widely respected 3-class sentiment model (negative, neutral, positive)
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    print("=" * 70)
    print("STAGE 1: PEEKING UNDER THE HOOD OF TEXT CLASSIFICATION")
    print("=" * 70)
    print(f"Loading Pretrained Model & Tokenizer: {model_name}...")

    # Load tokenizer and model separately to inspect both layers
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # Put model in inference mode (disables dropout)

    # Sample customer feedback to dissect
    sample_text = "The onboarding experience was seamless, but customer support took 48 hours to respond."
    print(f"\n[Raw Customer Review]:\n  \"{sample_text}\"")

    # ---------------------------------------------------------
    # STEP 1: TOKENIZATION
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: TOKENIZATION (Text -> Subwords -> Numerical IDs)")
    print("-" * 70)

    # 1.1 Show subword tokens (Byte-Pair Encoding)
    raw_tokens = tokenizer.tokenize(sample_text)
    print(f"1. Subword Tokens ({len(raw_tokens)} tokens):")
    print(f"   {raw_tokens}")
    print("   -> Notice how words like 'onboarding' or punctuation might be split into subwords.")
    print("   -> In RoBERTa, the special character 'Ġ' represents a leading space.")

    # 1.2 Convert to PyTorch tensors (input_ids and attention_mask)
    encoded_inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    print(f"\n2. input_ids Tensor (Shape: {encoded_inputs['input_ids'].shape}):")
    print(f"   {encoded_inputs['input_ids'][0].tolist()}")
    print("   -> Token ID 0 is <s> (start of sequence); Token ID 2 is </s> (end of sequence).")

    print(f"\n3. attention_mask Tensor (Shape: {encoded_inputs['attention_mask'].shape}):")
    print(f"   {encoded_inputs['attention_mask'][0].tolist()}")
    print("   -> '1' means attend to this real token. '0' would mean ignore padding.")

    # ---------------------------------------------------------
    # STEP 2: FORWARD PASS & RAW LOGITS
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: MODEL FORWARD PASS -> RAW LOGITS")
    print("-" * 70)

    with torch.no_grad():  # No need to compute gradients during inference
        outputs = model(**encoded_inputs)
        raw_logits = outputs.logits

    print(f"Raw Output Tensor (Logits) [Shape: {raw_logits.shape}]:")
    print(f"   {raw_logits[0].tolist()}")
    print("   -> Note: Logits are unconstrained real numbers (-inf to +inf).")
    print("   -> They are NOT probabilities! They do not sum to 1.0.")

    # ---------------------------------------------------------
    # STEP 3: THE SOFTMAX FUNCTION
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 3: SOFTMAX TRANSFORMATION (Logits -> Probabilities)")
    print("-" * 70)

    # Softmax formula: e^(z_i) / sum(e^(z_j))
    probabilities = torch.softmax(raw_logits, dim=-1)[0]
    prob_list = probabilities.tolist()

    print("Transformed Probabilities:")
    for idx, prob in enumerate(prob_list):
        label_name = model.config.id2label[idx]
        print(f"   Class {idx} ({label_name:>8}): {prob:.4f} ({prob*100:.2f}%)")

    sum_probs = sum(prob_list)
    print(f"\n   Sum of all probabilities: {sum_probs:.6f} (Guaranteed to sum to 1.0)")

    # ---------------------------------------------------------
    # STEP 4: INTERPRETING LABELS & CONFIDENCE
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 4: INTERPRETING PREDICTION & CONFIDENCE")
    print("-" * 70)

    top_class_id = torch.argmax(probabilities).item()
    top_label = model.config.id2label[top_class_id]
    top_confidence = probabilities[top_class_id].item()

    print(f"Final Model Prediction: {top_label.upper()}")
    print(f"Confidence Score:       {top_confidence:.4f} ({top_confidence*100:.2f}%)")

    # ---------------------------------------------------------
    # STEP 5: VERIFICATION WITH HIGH-LEVEL PIPELINE
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 5: VERIFYING AGAINST HUGGING FACE pipeline()")
    print("-" * 70)

    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    pipeline_result = pipe(sample_text)

    print(f"pipeline() Output: {pipeline_result}")
    print("\nObservation:")
    print(f"  Manual Calculation:  Label = {top_label}, Score = {top_confidence:.4f}")
    print(f"  pipeline() Result:   Label = {pipeline_result[0]['label']}, Score = {pipeline_result[0]['score']:.4f}")
    print("  -> They are mathematically identical! You now understand what pipeline() does inside.")
    print("=" * 70)

if __name__ == "__main__":
    run_stage_1_lab()

