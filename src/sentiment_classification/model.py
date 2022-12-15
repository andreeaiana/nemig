import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines.base import KeyDataset

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class SentimentClassificationModel():
    def __init__(
            self,
            ds_conf,
            overwrite_cache: bool,
            pretrained_model_path: str,
            padding: bool,
            truncation: bool,
            max_length: int,
            batch_size: int,
            sentiment_label_col: str,
            results_file: str
            ) -> None:
        
        self.ds_conf = ds_conf 
        self.overwrite_cache = overwrite_cache

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path)
        self.tokenizer_kwargs = {
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length
                }
        self.pipeline = pipeline(
                task = "sentiment-analysis",
                model = model,
                tokenizer = tokenizer,
                framework = "pt",
                batch_size = batch_size,
                device = 0
                )

        self.sentiment_label_col = sentiment_label_col
        self.results_file = Path(results_file)

    def classify_articles(self) -> None:
        if not self.overwrite_cache and os.path.isfile(self.results_file):
            log.info(f"Sentiment annotations already cached at {self.results_file}.")
        else:
            # data loading
            dataframe = pd.read_csv(self.ds_conf.data_processed_file)
            dataset = Dataset.from_pandas(
                    dataframe[[
                        self.ds_conf.id_col,
                        self.ds_conf.title_col,
                        self.ds_conf.description_col,
                        self.ds_conf.body_col
                        ]]
                    )
            content_column = [' '] * len(dataset)
            dataset = dataset.add_column("content", content_column)
            
            # concatenate input for sentiment analysis
            dataset = dataset.map(self._join_texts)

            # sentiment classification
            label_column = [output["label"] for output in tqdm(
                self.pipeline(KeyDataset(dataset, "content"), **self.tokenizer_kwargs), total = len(dataset)
                )]
            dataset = dataset.add_column(self.sentiment_label_col, label_column)
            dataset = dataset.remove_columns([
                self.ds_conf.title_col,
                self.ds_conf.description_col,
                self.ds_conf.body_col,
                "content"
                ])

            # number of examples per class 
            labels_count = Counter(dataset[self.sentiment_label_col])
            log.info(f"Number of articles per class: {labels_count}")

            # cache to disk
            dataset.to_csv(self.results_file, index = None)

    def _join_texts(self, article: pd.Series) -> pd.Series:
       article["content"] = ". ".join(
               [article[self.ds_conf.title_col], article[self.ds_conf.description_col]]
               ) if article[self.ds_conf.description_col] != None else ". ".join([article[self.ds_conf.title_col], article[self.ds_conf.body_col]])
      
       return article
