import torch
from transformers import BertModel, BertTokenizer, BertConfig
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import pandas as pd

dataset = pd.read_csv("SemEval2018-Task3/datasets/train/SemEval2018-T3-train-taskB.txt",sep='\t',header=0,index_col=0,quoting=csv.QUOTE_NONE, error_bad_lines=False)
tweets = dataset['Tweet text'][:20]

# Initialize bert
MODEL_NAME = 'bert-base-uncased'
config = BertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
model = BertModel.from_pretrained(MODEL_NAME, config=config)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


tweet_vectors = []
all_tokens = []
all_token_ids = []

for tweet in tweets:
    # Use the bert tokenizer\n",
    tokens = [tokenizer.cls_token] + tokenizer.tokenize(tweet) + [tokenizer.sep_token]
    all_tokens.append(tokens)
    print(tokens)

    # Convert the tokens to token ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    all_token_ids.append(token_ids)
    print(token_ids)
    print()
    tokens_tensor = torch.tensor(token_ids).unsqueeze(0)

    # Get the bert output, see https://huggingface.co/transformers/model_doc/bert.html#bert-specific-outputs for details
    model.eval()  # turn off dropout layers
    output = model(tokens_tensor)

    # Extract the hidden states for all layers
    layers = output.hidden_states

    # Select a layer to examine TODO: Try out different layers. For the assignment, it should be the last one. 
    layer = 12

    # Our batch consists of a single sentence, so we simply extract the first one
    tweet_vector = layers[layer][0].detach().numpy()

    # The sentence vector is a list of vectors, one for each token in the sentence
    # Each token vector consists of 768 dimensions
      
    # We use the vector for the first token (the CLS token) as representation for the sentence
    tweet_vectors.append(tweet_vector[0])
  

# Calculate pairwise similarities
similarity_matrix = cosine_similarity(np.asarray(tweet_vectors))
print(similarity_matrix)

# Plot a heatmap
ax = sns.heatmap(similarity_matrix, linewidth=0.5, cmap="YlGnBu")
ids = list(range(1, len(tweets)+1))
ax.set_xticklabels(ids)
ax.set_yticklabels(ids)

# Remove the ticks, but keep the labels
ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labeltop=True, labelbottom=False)
ax.set_title("Similarity between sentence pairs")
plt.show()
