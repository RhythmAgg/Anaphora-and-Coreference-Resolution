import gender_guesser.detector as gender
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Create a gender detector object
detector = gender.Detector()

def guess_gender(name):
  """Uses gender-guesser to estimate gender and probability.

  Args:
      name: The name to guess the gender for.

  Returns:
      A dictionary containing the estimated gender ("male", "female", 
      "andy" (ambiguous), or "unknown") and the probability (0.0 to 1.0).
  """
  result = detector.get_gender(name)
  gender_type, probability = result[0], result[1]
  return {"gender": gender_type, "probability": probability}

# Load English language model
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """
    Preprocess the input text using SpaCy.
    """
    doc = nlp(text)
    return doc

def exact_match_resolution(doc):
    """
    Resolve coreferences using exact string matches.
    """
    clusters = {}
    for mention in doc.ents:
        if mention.text not in clusters:
            clusters[mention.text] = [mention.text]
    return clusters

def strict_head_match_resolution(doc):
    """
    Resolve coreferences using strict head word matches.
    """
    clusters = {}
    for token in doc:
        if token.dep_ in ["nsubj", "dobj"]:  # Example dependency relations for head words
            head_word = token.head.text
            if head_word not in clusters:
                clusters[head_word] = [token.text]
            else:
                clusters[head_word].append(token.text)
    return clusters

def relaxed_head_match_resolution(doc, window_size=3):
    """
    Resolve coreferences using relaxed head word matches within a window size.
    """
    clusters = {}
    for token in doc:
        if token.dep_ in ["nsubj", "dobj"]:  # Example dependency relations for head words
            head_word = token.head.text
            start = max(0, token.i - window_size)
            end = min(len(doc), token.i + window_size + 1)
            for nearby_token in doc[start:end]:
                if nearby_token.text not in clusters:
                    clusters[nearby_token.text] = [token.text]
                else:
                    clusters[nearby_token.text].append(token.text)
    return clusters

def string_match_resolution(doc):
    """
    Resolve coreferences based on string similarity.
    """
    clusters = {}
    for mention in doc.ents:
        for cluster_head in clusters:
            if mention.text.lower() == cluster_head.lower():
                clusters[cluster_head].append(mention.text)
                break
    return clusters

def proper_head_noun_match_resolution(doc):
    """
    Resolve coreferences using proper head nouns.
    """
    clusters = {}
    for token in doc:
        if token.pos_ == "PROPN":
            if token.text not in clusters:
                clusters[token.text] = [token.text]
    return clusters

def appositive_match_resolution(doc):
    """
    Resolve coreferences based on appositive relationships.
    """
    clusters = {}
    for token in doc:
        if token.dep_ == "appos":
            antecedent = token.head.text
            if antecedent in clusters:
                clusters[antecedent].append(token.text)
            else:
                clusters[antecedent] = [token.text]
    return clusters

def overlap_match_resolution(doc):
    """
    Resolve coreferences based on word overlap.
    """
    clusters = {}
    for mention in doc.ents:
        for cluster_head in clusters:
            overlap = set(mention.text.lower().split()).intersection(set(cluster_head.lower().split()))
            if overlap:
                clusters[cluster_head].append(mention.text)
                break
    return clusters

def pronoun_resolution(doc, clusters):
    """
    Resolve coreferences using pronouns.
    """
    for token in doc:
        if token.pos_ == "PRON":
            antecedent = None
            for cluster in clusters.values():
                for mention in cluster:
                    if token.text.lower() in ["he", "him", "his"] and guess_gender(mention)['gender'] == 'm':
                        antecedent = mention
                        break
                    elif token.text.lower() in ["she", "her", "hers"] and guess_gender(mention)['gender'] == 'f':
                        antecedent = mention
                        break
                    # Add more rules for other pronouns and antecedents
            if antecedent:
                if antecedent in clusters:
                    clusters[antecedent].append(token.text)
                else:
                    clusters[antecedent] = [token.text]
    return clusters

# Load pre-trained word embeddings (GloVe)
def load_word_embeddings(embeddings_path):
    embeddings_index = {}
    with open(embeddings_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index

def semantic_similarity_resolution(doc, clusters, embeddings_index):
    """
    Resolve coreferences using semantic similarity.
    """
    for mention in doc.ents:
        if mention.text not in clusters:
            for cluster_head in clusters:
                # Compute cosine similarity between mention and cluster head
                if cluster_head in embeddings_index and mention.text in embeddings_index:
                    similarity = cosine_similarity([embeddings_index[cluster_head]], [embeddings_index[mention.text]])
                    # Threshold for considering two mentions as the same entity
                    if similarity > 0.7:  # Adjust threshold as needed
                        clusters[cluster_head].append(mention.text)
                        break
    return clusters

def resolve_coreferences(text, embeddings_index=None):
    """
    Resolve coreferences using multi-pass sieve method.
    """
    # Preprocess the input text
    doc = preprocess(text)
    
    # Pass 1: Exact Match
    clusters = exact_match_resolution(doc)
    
    # Pass 2: Strict Head Match
    strict_head_clusters = strict_head_match_resolution(doc)
    for head, mentions in strict_head_clusters.items():
        if head not in clusters:
            clusters[head] = mentions
        else:
            clusters[head].extend(mentions)
    
    # Pass 3: Relaxed Head Match
    relaxed_head_clusters = relaxed_head_match_resolution(doc)
    for head, mentions in relaxed_head_clusters.items():
        if head not in clusters:
            clusters[head] = mentions
        else:
            clusters[head].extend(mentions)
    
    # Pass 4: Pronoun Resolution
    clusters = pronoun_resolution(doc, clusters)
    
    # Pass 5: Semantic Similarity
    if embeddings_index:
        clusters = semantic_similarity_resolution(doc, clusters, embeddings_index)
    
    return clusters

# Example usage
input_text = "John went to the store. He bought some groceries. Jessica is his best friend and she likes him a lot. She is very beautiful."
embeddings_index = load_word_embeddings("/kaggle/input/glove6b100dtxt/glove.6B.100d.txt")
coreference_clusters = resolve_coreferences(input_text, embeddings_index)
print(coreference_clusters)
