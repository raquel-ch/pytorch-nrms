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

from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

from ebrec.models.newsrec_pytorch.dataloader import NRMSDataLoader
from ebrec.models.newsrec_pytorch.model_config import hparams_nrms
from ebrec.models.newsrec_pytorch import NRMSModel

# %% [markdown]
# ## Load dataset

# %%
def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
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
DATASPLIT = "ebnerd_demo"
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
HISTORY_SIZE = 10
FRACTION = 0.01

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
df_articles = pl.read_parquet(PATH.joinpath("ebnerd_demo/articles.parquet"))
df_articles.head(2)

# %% [markdown]
# ## Init model using HuggingFace's tokenizer and wordembedding
# In the original implementation, they use the GloVe embeddings and tokenizer. To get going fast, we'll use a multilingual LLM from Hugging Face. 
# Utilizing the tokenizer to tokenize the articles and the word-embedding to init NRMS.
# 

# %%
TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30

# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

# We'll init the word embeddings using the
word2vec_embedding = get_transformers_word_embeddings(transformer_model)
#
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)

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
    batch_size=4,
)
val_dataloader = NRMSDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=4,
)

# %% [markdown]
# ## Train the model
# 

# %%
import torch
import torch.nn as nn
epoch = 0
num_epochs = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f">> Using device: {device}")
from ebrec.models.newsrec_pytorch import NRMSModel
from torch.utils.tensorboard import SummaryWriter


word2vec_embedding = torch.tensor(word2vec_embedding, dtype=torch.float32).to(device)
print(word2vec_embedding.shape)
nrms = NRMSModel(hparams_nrms=hparams_nrms, word2vec_embedding=word2vec_embedding, seed=42).to(device) # Adding to device
print(nrms)
optimizer = torch.optim.Adam(nrms.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter("./logs")

# Training loop
for epoch in range(num_epochs):
    nrms.train()  # Set the model to training mode
    running_loss = 0.0
    for i, ((his_input_title, pred_input_title), labels) in enumerate(train_dataloader):
        his_input_title = his_input_title.to(device, dtype=torch.long)
        pred_input_title = pred_input_title.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.long).view(-1)
        labels = labels.view(-1)
        # print("Input shape for user encoder:", his_input_title.shape)
        # print("Input shape for news encoder:", pred_input_title.shape)
        optimizer.zero_grad()  # Zero the gradients
        outputs = nrms(his_input_title, pred_input_title)  # Forward pass
        # print(outputs)
        loss = loss_fn(outputs.view(-1), labels.float())  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the parameters
        running_loss += loss.item()
        
        
        # Detach tensors immediately after use to save memory
        his_input_title = his_input_title.detach()
        pred_input_title = pred_input_title.detach()
        labels = labels.detach()
        outputs = outputs.detach()
        del his_input_title, pred_input_title, labels, outputs, loss
        torch.cuda.empty_cache()  # Clear unused GPU memory
        
        
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0
    print("Validation started ----")
    # Validation
    nrms.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for (his_input_title, pred_input_title), labels in val_dataloader:
            his_input_title = his_input_title.to(device, dtype=torch.long)
            pred_input_title = pred_input_title.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long).view(-1)
            labels = labels.view(-1)
            
            outputs = nrms(his_input_title, pred_input_title)
            loss = loss_fn(outputs.view(-1), labels.float())
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Detach tensors immediately after use to save memory
            his_input_title = his_input_title.detach()
            pred_input_title = pred_input_title.detach()
            labels = labels.detach()
            outputs = outputs.detach()
            del his_input_title, pred_input_title, labels, outputs, loss
            torch.cuda.empty_cache()  # Clear unused GPU memory
            
    val_loss /= len(val_dataloader)
    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}, Loss: {val_loss:.3f}, Accuracy: {accuracy:.2f}%")
    
       # Log to TensorBoard
    writer.add_scalar("Loss/train", running_loss, epoch)
    writer.add_scalar("Loss/validation", val_loss, epoch)
    writer.add_scalar("Accuracy/validation", accuracy, epoch)

# Save the model weights
# MODEL_NAME = "NRMS"
# MODEL_WEIGHTS = f"downloads/data/state_dict/{MODEL_NAME}/weights"
# torch.save(nrms.state_dict(), MODEL_WEIGHTS)
            
        
    

# %%
# MODEL_NAME = "NRMS"
# LOG_DIR = f"downloads/runs/{MODEL_NAME}"
# MODEL_WEIGHTS = f"downloads/data/state_dict/{MODEL_NAME}/weights"

# # CALLBACKS
# # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
# # early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
# # modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
# #     filepath=MODEL_WEIGHTS, save_best_only=True, save_weights_only=True, verbose=1
# # )

# hparams_nrms.history_size = HISTORY_SIZE


# model = NRMSModel(
#     hparams=hparams_nrms,
#     word2vec_embedding=word2vec_embedding,
#     seed=42,
# )
# hist = model.model.fit(
#     train_dataloader,
#     validation_data=val_dataloader,
#     epochs=1,
#     # callbacks=[tensorboard_callback, early_stopping, modelcheckpoint],
# )
# Uncomment the following line if you have pre-trained weights
# _ = model.model.load_weights(filepath=MODEL_WEIGHTS)

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


