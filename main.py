import streamlit as st
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd

# Load model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("ner_finetuned_model")
tokenizer = AutoTokenizer.from_pretrained("ner_finetuned_model")

# Function to map predicted labels to entity names
id_to_label = {0: 'PERSON', 1: 'ORG', 2: 'LOC', 3: 'O', 4: 'DATE'}


# Batch NER Detection Function
def ner_detection_batch(sentences):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = []
    for text in sentences:
        # Tokenize the input text and move tokens to the same device
        tokens = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False).to(device)

        with torch.no_grad():
            output = model(**tokens)

        # Get predictions
        predictions = np.argmax(output.logits.detach().cpu().numpy(), axis=2)

        # Convert token IDs to words and labels
        tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze().tolist())
        labels = [id_to_label[label] for label in predictions[0]]

        # Combine tokens and their predicted labels
        sentence_results = []
        word = ""
        word_label = ""

        for token, label in zip(tokens, labels):
            # Skip special tokens like [CLS], [SEP], [PAD]
            if token.startswith("##"):  # Part of a subword token
                word += token[2:]  # Remove the "##" prefix and append to the current word
            elif token not in ["[CLS]", "[SEP]", "[PAD]", ".", ",", ":", ";", "!", "?"]:  # Skip punctuation
                if word:  # If there is a word being built, add it with the previous label
                    sentence_results.append((word, word_label))
                    word = ""  # Reset for the next word
                sentence_results.append((token, label))
                word_label = label
            else:
                # Continue to accumulate the word if it's a valid token
                if word:
                    sentence_results.append((word, word_label))
                    word = ""  # Reset for the next word

        # If the last word is being built
        if word:
            sentence_results.append((word, word_label))

        results.append(sentence_results)

    return results


# Streamlit app
st.title("Named Entity Recognition (NER) with Fine-tuned Model")

# User input
user_input = st.text_area("Enter a sentence or multiple sentences:")

# Process the input and show results
if st.button("Detect"):
    sentences = user_input.split("\n")
    detected_entities_batch = ner_detection_batch(sentences)

    # Convert results into a pandas DataFrame for better display
    data = []
    for sentence_idx, entities in enumerate(detected_entities_batch):
        for token, label in entities:
            data.append([f"Sentence {sentence_idx + 1}", token, label])

    df = pd.DataFrame(data, columns=["Sentence", "Token", "Predicted Entity"])

    # Display the results
    if len(df) > 0:
        st.write("### Detected Named Entities")
        st.dataframe(df)  # Displays the dataframe in a table format
    else:
        st.write("No entities detected.")
