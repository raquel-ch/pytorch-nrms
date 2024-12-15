#!/usr/bin/env python3
import argparse
# %% [markdown]
# # Getting started
# 
# In this notebook, we illustrate how to use the Neural News Recommendation with Multi-Head Self-Attention ([NRMS](https://aclanthology.org/D19-1671/)). The implementation is taken from the [recommenders](https://github.com/recommenders-team/recommenders) repository. We have simply stripped the model to keep it cleaner.
# 
# We use a small dataset, which is downloaded from [recsys.eb.dk](https://recsys.eb.dk/). All the datasets are stored in the folder path ```~/ebnerd_data/*```.

# %%
import torch
import torch.nn as nn
epoch = 0
num_epochs = 10
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")

# %% [markdown]
# ## Load functionality

# %%
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import polars as pl
from torch.utils.tensorboard import SummaryWriter


from _constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)

from _behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
from _articles import convert_text2encoding_with_transformers
from _polars import concat_str_columns, slice_join_dataframes
from _articles import create_article_id_to_value_mapping
from _nlp import get_transformers_word_embeddings
from _python import write_submission_file, rank_predictions_by_score

from dataloader import NRMSDataLoader
from model_config import hparams_nrms
from NRMSModel import NRMSModel

# %% [markdown]
# ## Load dataset

# %%
def get_parameters():
    learning_rate = float(input("Learning rate: "))
    batch_size = int(input("Batch size: "))
    epochs = int(input("Epochs: "))
    weight_decay = float(input("Weight decay: "))
    head_dim = int(input("Head dimension/number: "))
    history_size = int(input("History size: "))
    
    return learning_rate, batch_size, epochs, weight_decay, head_dim, history_size

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("lr", type=float)
parser.add_argument("bs", type=int)
parser.add_argument("ep", type=int)
parser.add_argument("wd", type=float)
parser.add_argument("head", type=int)
parser.add_argument("hs", type=int)

args = parser.parse_args()

# Add parsed values
learning_rate = args.lr
batch_size = args.bs
epochs = args.ep
weight_decay = args.wd
head_dim = args.head
history_size = args.hs

#learning_rate, batch_size, epochs, weight_decay, head_dim, history_size = get_parameters()
hparams_nrms.learning_rate = learning_rate
hparams_nrms.batch_size = batch_size
hparams_nrms.epochs = epochs
hparams_nrms.weight_decay = weight_decay
hparams_nrms.head_dim = head_dim
hparams_nrms.head_num = head_dim
hparams_nrms.history_size = history_size

def ebnerd_from_path(path: Path, history_size: int = 10) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",
        )
    )
    return df_behaviors

# %% [markdown]
# ### Generate labels
# We sample a few just to get started. For testset we just make up a dummy column with 0 and 1 - this is not the true labels.

# %%
PATH = Path("./ebnerd_data").expanduser()
DATASPLIT = "ebnerd_small"
DUMP_DIR = PATH.joinpath("downloads1")
DUMP_DIR.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# In this example we sample the dataset, just to keep it smaller. Also, one can simply add the testset similary to the validation.

# %% [markdown]
# 

# %%
COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]
HISTORY_SIZE = hparams_nrms.history_size
FRACTION = 0.2

df_train = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=123,
    )
    .pipe(create_binary_labels_column)
    .sample(fraction=FRACTION)
)
# =>
df_validation = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(create_binary_labels_column)
    .sample(fraction=FRACTION)
)
df_train.head(2)

# %% [markdown]
# ## Load articles

# %%
df_articles = pl.read_parquet(PATH.joinpath("ebnerd_small/articles.parquet"))
df_articles.head(2)

# %% [markdown]
# ## Init model using HuggingFace's tokenizer and wordembedding
# In the original implementation, they use the GloVe embeddings and tokenizer. To get going fast, we'll use a multilingual LLM from Hugging Face. 
# Utilizing the tokenizer to tokenize the articles and the word-embedding to init NRMS.
# 

# %%
TRANSFORMER_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30

# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

# We'll init the word embeddings using the
word2vec_embedding = get_transformers_word_embeddings(transformer_model)
print(word2vec_embedding.shape)
#
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)

# %%
# try to check if files exist
import os
import numpy as np
if not os.path.exists('vocab_npa.npy') or not os.path.exists('embs_npa.npy'):
    vocab,embeddings = [],[]
    with open('glove.6B.300d.txt','rt') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
        
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)

    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    print(vocab_npa[:10])

    pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

    #insert embeddings for pad and unk tokens at top of embs_npa.
    embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))

    with open('vocab_npa.npy','wb') as f:
        np.save(f,vocab_npa)

    with open('embs_npa.npy','wb') as f:
        np.save(f,embs_npa)

else:
    embs_npa = np.load('embs_npa.npy')
    vocab_npa = np.load('vocab_npa.npy')

embs_npa = torch.tensor(embs_npa).float()
print(embs_npa.shape)
    

# %% [markdown]
# # Initiate the dataloaders
# In the implementations we have disconnected the models and data. Hence, you should built a dataloader that fits your needs.

# %%
train_dataloader = NRMSDataLoader(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=hparams_nrms.batch_size,
)
val_dataloader = NRMSDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=hparams_nrms.batch_size,
)

# %%
def print_hparams(hparams):
    print("Hyperparameters:")
    print(f"Learning rate: {hparams.learning_rate}")
    print(f"Batch size: {hparams.batch_size}")
    print(f"Epochs: {hparams.epochs}")
    print(f"Weight decay: {hparams.weight_decay}")
    print(f"Head dimension: {hparams.head_dim}")
    print(f"Head number: {hparams.head_num}")
    
        
print_hparams(hparams_nrms)

# %% [markdown]
# ## Train the model
# 

# %%
from sklearn.metrics import roc_auc_score
import torch.nn.utils  # Ensure this is imported for gradient clipping

epoch = 0
num_epochs = hparams_nrms.epochs

word2vec_embedding = embs_npa.to(device)

nrms = NRMSModel(hparams_nrms=hparams_nrms, word2vec_embedding=word2vec_embedding, seed=50).to(device)  # Adding to device
print(nrms)

for name, param in nrms.named_parameters():
    print(f"Parameter: {name}, Requires Grad: {param.requires_grad}, Shape: {param.shape}")


optimizer = torch.optim.Adam(nrms.parameters(), lr=hparams_nrms.learning_rate, weight_decay=hparams_nrms.weight_decay)
loss_fn = nn.CrossEntropyLoss()
val_loss_fn = nn.CrossEntropyLoss()

# Gradient clipping parameter
max_norm = 5.0  # Maximum gradient norm
running_losses = []
validation_losses = []
training_aucs = []
validation_aucs = []
# Training loop
for epoch in range(num_epochs):
    nrms.train()  # Set the model to training mode
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    
    for i, ((his_input_title, pred_input_title), labels) in enumerate(train_dataloader):
        his_input_title = his_input_title.to(device, dtype=torch.long)
        pred_input_title = pred_input_title.to(device, dtype=torch.long)
        og_labels = labels
        labels = labels.to(device, dtype=torch.long).view(-1)

        optimizer.zero_grad()  # Zero the gradients
        outputs = nrms(his_input_title, pred_input_title).to(device)  # Forward pass
        
        loss = loss_fn(outputs.view(-1), labels.float())  # Compute the loss
        loss.backward()  # Backward pass
        loss.detach()  # Detach the loss to save memory
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(nrms.parameters(), max_norm)

        optimizer.step()  # Update the parameters
        running_loss += loss.item()
    
        
        # Detach tensors immediately after use to save memory
        his_input_title.detach()
        pred_input_title.detach()
        labels.detach()
        
        # Save labels and outputs for AUC calculation
        all_labels.extend(og_labels.detach().cpu().numpy())
        all_outputs.extend(outputs.detach().cpu().numpy())
        
        del his_input_title, pred_input_title, labels, loss, outputs, og_labels
        torch.cuda.empty_cache()  # Clear unused GPU memory

    running_loss /= len(train_dataloader)
    running_losses.append(running_loss)
    # Calculate AUC score
    auc = 0
    for i, label_true in enumerate(all_labels):
        auc += roc_auc_score(label_true, all_outputs[i])
    auc /= len(all_labels)
    training_aucs.append(auc)
    
    # Print training details
    print(f"Epoch: {epoch + 1}/{num_epochs}")
    print(f"Training loss: {running_loss:.10f}, Training AUC: {auc:.10f}")
    #print(f"Training outputs: {all_outputs[:10]}")
    #print(f"Training labels: {all_labels[:10]}")

    # Write training AUC values to file
    with open('outputtest.txt', 'a') as f:
        f.write(f"(Tr) Epoch: {epoch}, AUC: {auc}\n")

    # Validation loop
    nrms.eval()  # Set the model to evaluation mode
    all_labels = []
    all_outputs = []
    val_loss = 0.0
    with torch.no_grad():
        for i, ((his_input_title, pred_input_title), labels) in enumerate(val_dataloader):
            his_input_title = his_input_title.to(device, dtype=torch.long)
            pred_input_title = pred_input_title.to(device, dtype=torch.long)
            og_labels = labels
            labels = labels.to(device, dtype=torch.long).view(-1)

            outputs = nrms(his_input_title, pred_input_title).to(device)  # Forward pass
            loss = val_loss_fn(outputs.view(-1), labels.float())
            val_loss += loss.item()
            
            all_labels.extend(og_labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            
            # Detach tensors immediately after use to save memory
            his_input_title = his_input_title.detach()
            pred_input_title = pred_input_title.detach()
            labels = labels.detach()
            outputs = outputs.detach()
            del his_input_title, pred_input_title, labels, outputs
            torch.cuda.empty_cache()
            
    val_loss /= len(val_dataloader)
    validation_losses.append(val_loss)

    # Calculate AUC score
    auc = roc_auc_score(all_labels, all_outputs)
    validation_aucs.append(auc)
    
    # print(f"Validation outputs: {all_outputs[:10]}")
    # print(f"Validation labels: {all_labels[:10]}")
    
    # Print validation details
    print(f"Validation loss: {val_loss:.10f}, Validation AUC: {auc:.10f}")
    print(f"--------------------------\n")

    # Write validation AUC values to file
    with open('outputtest.txt', 'a') as f:
        f.write(f"(Val) Epoch: {epoch}, AUC: {auc}\n")
        
        



# %%
# Plot the training loss and validation loss
import matplotlib.pyplot as plt
print(running_losses)
print(validation_losses)
plt.title("Configuration 1 - Loss")
plt.plot(list(range(1, num_epochs + 1)), running_losses, label="Training Loss")
# steps for x-axis
plt.xticks(list(range(1, num_epochs + 1, 2)))
# steps for y-axis should be of 0.2 from 0 to 1
# plt.yticks([i/2 for i in range(0, 3)])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(range(1, num_epochs + 1), validation_losses, label="Validation Loss")
plt.legend()
plt.show()
plt.savefig(f"plots/loss_bs{hparams_nrms.batch_size}_lr{hparams_nrms.learning_rate}_wd{hparams_nrms.weight_decay}_hd{hparams_nrms.head_dim}_hn{hparams_nrms.head_num}_hs{hparams_nrms.history_size}.png")

print("Loss plot saved :D")

# %%
# Plot the training loss and validation loss
import matplotlib.pyplot as plt
# Reset the plot
plt.plot()
plt.title("Configuration 1 - AUC")
plt.plot(list(range(1, num_epochs + 1)), training_aucs, label="Training AUC")
# steps for x-axis
plt.xticks(list(range(1, num_epochs + 1, 2)))
# steps for y-axis should be of 0.05 from 0 to 1
# plt.yticks([i/20 for i in range(0, 21)])

plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.plot(range(1, num_epochs + 1), validation_aucs, label="Validation AUC")
plt.legend()
plt.show()
plt.savefig(f"plots/auc_bs{hparams_nrms.batch_size}_lr{hparams_nrms.learning_rate}_wd{hparams_nrms.weight_decay}_hd{hparams_nrms.head_dim}_hn{hparams_nrms.head_num}_hs{hparams_nrms.history_size}.png")

print("AUC plot saved :D")

# %% [markdown]
# # Example how to compute some metrics:

# %%
# pred_validation = model.scorer.predict(val_dataloader)

# %% [markdown]
# ## Add the predictions to the dataframe

# %%
# df_validation = add_prediction_scores(df_validation, pred_validation.tolist()).pipe(
#     add_known_user_column, known_users=df_train[DEFAULT_USER_COL]
# )
# df_validation.head(2)

# %% [markdown]
# ### Compute metrics

# %%
# metrics = MetricEvaluator(
#     labels=df_validation["labels"].to_list(),
#     predictions=df_validation["scores"].to_list(),
#     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
# )
# metrics.evaluate()

# %% [markdown]
# ## Make submission file

# %%
# df_validation = df_validation.with_columns(
#     pl.col("scores")
#     .map_elements(lambda x: list(rank_predictions_by_score(x)))
#     .alias("ranked_scores")
# )
# df_validation.head(2)

# %% [markdown]
# This is using the validation, simply add the testset to your flow.

# %%
# write_submission_file(
#     impression_ids=df_validation[DEFAULT_IMPRESSION_ID_COL],
#     prediction_scores=df_validation["ranked_scores"],
#     path="downloads/predictions.txt",
# )

# %% [markdown]
# # DONE 🚀


