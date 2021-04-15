import torch
from transformers import BertModel, BertTokenizer, BertConfig
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Initialize bert
MODEL_NAME = 'bert-base-uncased'
config = BertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
model = BertModel.from_pretrained(MODEL_NAME, config=config)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# TODO: Play around with some more sentences here.
sentences = ["I just love deadlines.",
             "Love is all around you.", "Love is in the air."]

sentence_vectors = []
all_tokens = []
all_token_ids = []

for sentence in sentences:
    # Use the bert tokenizer\n",
    tokens = [tokenizer.cls_token] + tokenizer.tokenize(sentence) + [tokenizer.sep_token]
    all_tokens.append(tokens)
    print(tokens)

    # Convert the tokens to token ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    all_token_ids.append(token_ids)
    print(token_ids)
    tokens_tensor = torch.tensor(token_ids).unsqueeze(0)

    # Get the bert output, see https://huggingface.co/transformers/model_doc/bert.html#bert-specific-outputs for details
    model.eval()  # turn off dropout layers
    output = model(tokens_tensor)

    # Extract the hidden states for all layers
    layers = output.hidden_states
    print(len(layers))

    # Select a layer to examine TODO: Try out different layers. For the assignment, it should be the last one. 
    layer = 3

    # Our batch consists of a single sentence, so we simply extract the first one
    sentence_vector = layers[layer][0].detach().numpy()

    # The sentence vector is a list of vectors, one for each token in the sentence
    # Each token vector consists of 768 dimensions
    print(sentence_vector.shape)
      
    # We use the vector for the first token (the CLS token) as representation for the sentence
    sentence_vectors.append(sentence_vector[0])
  

# Calculate pairwise similarities
similarity_matrix = cosine_similarity(np.asarray(sentence_vectors))
print(similarity_matrix)

# Plot a heatmap
ax = sns.heatmap(similarity_matrix, linewidth=0.5, cmap="YlGnBu")
ids = list(range(1, len(sentences)+1))
ax.set_xticklabels(ids)
ax.set_yticklabels(ids)

# Remove the ticks, but keep the labels
ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labeltop=True, labelbottom=False)
ax.set_title("Similarity between sentence pairs")
plt.show()
