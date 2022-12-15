from typing import List, Dict

import os
import stanza
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines.base import KeyDataset

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)
tqdm.pandas()


class NERModel():
    def __init__(
            self,
            ds_conf, 
            language: str,
            overwrite_cache: bool,
            columns2extract: Dict[str, str],
            columns2sentencize: List[str],
            columns_list_feat: List[str],
            pretrained_model_path: str,
            batch_size: int,
            aggregation_strategy: str,
            results_files: Dict[str, str],
            ) -> None:

        self.ds_conf = ds_conf
        self.overwrite_cache = overwrite_cache
        self.columns2extract = columns2extract
        self.columns2sentencize = columns2sentencize
        self.columns_list_feat= columns_list_feat
        self.results_files = results_files

        assert all([col in self.columns2extract.keys() for col in self.columns2sentencize])
        assert all([col in self.columns2extract.keys() for col in self.columns_list_feat])
        assert all([col in self.results_files.keys() for col in self.columns2extract.keys()])

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        model = AutoModelForTokenClassification.from_pretrained(pretrained_model_path)
        self.pipeline = pipeline(
                task = "ner",
                model = model,
                tokenizer = tokenizer,
                framework = "pt",
                batch_size = batch_size,
                device = 0,
                aggregation_strategy = aggregation_strategy
                )

        self.sentencizer = stanza.Pipeline(
                lang = language,
                processors = "tokenize",
                tokenize_batch_size = batch_size,
                use_gpu = True
                )

    def extract_entities(self) -> None:
        # data loading
        all_dataframe = pd.read_csv(self.ds_conf.data_processed_file)

        # extract named entities from each chosen column
        for col in self.columns2extract.keys():
            col2extract = self.columns2extract[col]

            if not self.overwrite_cache and os.path.isfile(self.results_files[col]):
                log.info(f"Named entity annotations for articles' {col2extract} already cached at {self.results_files[col]}.")
            
            else:
                log.info(f"Extracting named entities from the articles' {col2extract}.")
                dataframe = all_dataframe[[self.ds_conf.id_col, col2extract]]
                
                # drop rows with empty feature (e.g. no description)
                if col in self.columns_list_feat:
                    if col == 'topic_label':
                        dataframe[col2extract] = dataframe[col2extract].apply(lambda x: x.split('_'))
                    else:
                        dataframe[col2extract] = dataframe[col2extract].apply(literal_eval)
                    rows_no_feature = dataframe[dataframe[col2extract].apply(lambda x: len(x)==0)].index
                else:
                    rows_no_feature = dataframe[dataframe[col2extract].isnull()].index
                dataframe = dataframe.drop(rows_no_feature)

                # sentence segmentation for long texts (i.e. article body)
                if col in self.columns2sentencize:
                    dataframe[col2extract] = dataframe[col2extract].progress_apply(lambda paragraph: self._sentencize(paragraph))
                    
                    # explode dataframe to one sentence per row
                    dataframe = dataframe.explode(col2extract, ignore_index=True)

                # explode columns where the features are a list of strings (e.g. authors, keywords, topics)
                if col in self.columns_list_feat:
                    dataframe = dataframe.explode(col2extract, ignore_index=True)
                    dataframe[col2extract] = dataframe[col2extract].astype('str')

                dataset = Dataset.from_pandas(dataframe, preserve_index = False)

                # named entity recognition
                ner_results = [output for output in tqdm(
                    self.pipeline(KeyDataset(dataset, col2extract)), total = len(dataset)
                    )]
                dataset = dataset.add_column("ner_results", ner_results)

                # group 2-word entities (e.g. "U" -> LOC and "S."-> LOC into "US." -> LOC)
                dataset = dataset.map(self._group_entities)

                # remove invalid entities (i.e. 1-word entities)
                dataset = dataset.map(self._remove_invalid_entities)

                # postprocess dataset
                ner_data = dataset.to_pandas()
                ner_data = ner_data.explode("ner_results", ignore_index = True)
                
                ner_data = pd.concat(
                        [ner_data.drop(["ner_results"], axis=1), ner_data["ner_results"].apply(pd.Series)],
                        axis=1
                        )
                
                # remove rows (e.g. titles, sentences) without named entities
                if col2extract not in self.columns_list_feat:
                    rows2drop = ner_data[ner_data["word"].isnull()].index
                    ner_data = ner_data.drop(rows2drop)

                # remove extra column created when NaN values are in the dataframe
                if 0 in ner_data.columns:
                    ner_data = ner_data.drop(columns=[0])

                # cache to disk
                ner_data = ner_data.reset_index(drop=True)
                ner_data.to_csv(self.results_files[col], index = None)

    def _sentencize(self,  paragraph: str) -> List[str]:
        # Avoids RuntimeError for one single document
        try:
            return [sentence.text for sentence in self.sentencizer(paragraph).sentences]
        except RuntimeError:
            return paragraph.split('. ')

    def _group_entities(self, article: pd.Series) -> pd.Series:
        entities = []

        for result in article["ner_results"]:
            if len(entities) >= 1:
                previous_token = entities[len(entities)-1]["word"]
                previous_end_position = entities[len(entities)-1]["end"]
                previous_entity_group = entities[len(entities)-1]["entity_group"]

                current_token = result["word"]
                current_start_position = result["start"]
                current_entity_group = result["entity_group"]

                if (len(previous_token) <= 2 and len(current_token) <= 2) and (previous_entity_group == current_entity_group):
                    if current_start_position == previous_end_position + 1:
                        new_result = {
                                "word": "".join([previous_token, current_token]),
                                "entity_group": current_entity_group,
                                "start": entities[len(entities)-1]["start"],
                                "end": result["end"],
                                "score": 0.5 * (entities[len(entities)-1]["score"] + result["score"])
                                }

                        entities.pop()
                        entities.append(new_result)
                else:
                    entities.append(result)
            else:
                entities.append(result)

        article["ner_results"] = entities

        return article

    def _remove_invalid_entities(self, article: pd.Series) -> pd.Series:
        valid_entities = [ent for ent in article["ner_results"] if len(ent["word"]) > 1]
        article["ner_results"] = valid_entities

        return article
