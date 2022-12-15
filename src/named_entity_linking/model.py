from typing import Any, List, Dict, Iterable

import os
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

from src.named_entity_linking.genre.trie import Trie
from src.named_entity_linking.genre.hf_model import mGENRE
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class NELModel():
    def __init__(
            self,
            overwrite_cache: bool,
            pretrained_model_path: str,
            lang_title2wikidataID_path: str,
            trie_path: str,
            batch_size: int,
            columns2link: Dict[str, str],
            columns_single_entities: List[str],
            input_data_files: Dict[str, str],
            results_files: Dict[str, str]
            ) -> None:
       
        self.overwrite_cache = overwrite_cache
        self.batch_size = batch_size
        self.columns2link = columns2link
        self.columns_single_entites = columns_single_entities
        self.input_data_files = input_data_files
        self.results_files = results_files

        assert all([col in self.columns2link.keys() for col in columns_single_entities])
        assert all([col in self.input_data_files.keys() for col in self.columns2link.keys()])
        assert all([col in self.results_files.keys() for col in self.columns2link.keys()])

        # Initialize NEL model
        log.info("Initializing mGENRE model")
        self.model = mGENRE.from_pretrained(pretrained_model_path) 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        log.info("Loading strings to Wikidata IDs mapping")
        with open(lang_title2wikidataID_path, "rb") as f:
            self.lang_title2wikidataID = pickle.load(f)

        log.info("Loading prefix tree")
        with open(trie_path, "rb") as f:
            self.trie = Trie.load_from_dict(pickle.load(f))

    def link_entities(self) -> None:

        # extact named entities for each chosen column
        for col in self.columns2link.keys():

            self.col2link = self.columns2link[col]

            if not self.overwrite_cache and os.path.isfile(self.results_files[col]):
                log.info(f"Linked entity annotations for articles' {self.col2link} already cached at {self.results_files[col]}.")
            
            else:
                log.info(f"Link entities from the articles' {self.col2link}")

                # data loading
                try:
                    dataframe = pd.read_csv(self.input_data_files[col])
                    
                    # drop rows without entities
                    empty_rows = dataframe[dataframe[self.col2link].isnull()].index
                    dataframe = dataframe.drop(empty_rows)
                    dataframe = dataframe.reset_index(drop=True)

                    dataset = Dataset.from_pandas(dataframe, preserve_index=False)

                    # add special tokens at the beginning and end of an entity
                    if self.col2link not in self.columns_single_entites:
                        dataset = dataset.map(self._add_special_tokens_text)
                    else:
                        dataset = dataset.map(self._add_special_tokens_single_entities)
                
                    dataset = dataset.rename_column("score", "ner_score")

                    # named entity linking
                    batches = self._batchify(dataset)
                    count_batches = int(len(dataset) / self.batch_size) + 1
                    nel_results = []
                    
                    for batch in tqdm(batches, total=count_batches):
                        
                        # quick fix for too long sentences (all entities contained within the first part of the text)
                        sentences = [sentence if len(sentence)<=2500 else sentence[:2500] for sentence in batch[self.col2link]]

                        batch_outputs = self.model.sample(
                                sentences,
                                num_return_sequences=1,
                                prefix_allowed_tokens_fn = lambda _, sent: [
                                    e for e in self.trie.get(sent.tolist()) if e < len(self.model.tokenizer) - 1
                                    ],
                                text_to_id = lambda x: max(
                                    self.lang_title2wikidataID[tuple(reversed(x.split(" >> ")))], 
                                    key=lambda y: int(y[1:])) if tuple(reversed(x.split(" >> "))) in self.lang_title2wikidataID.keys() else None,
                                )
                        nel_results.extend([output[0] for output in batch_outputs])

                    nel_results = [
                            {"linked_entity": item['text'].rsplit(" >> ")[0],
                             "linked_entity_language": item["text"].rsplit(" >> ")[-1],
                             "nel_score": float(item["score"]),
                             'wikidata_id': item['id']
                             }
                            for item in nel_results
                            ]
                    dataset = dataset.add_column("nel_results", nel_results)

                    # postprocess dataset
                    nel_data = dataset.to_pandas()
                    nel_data = pd.concat(
                            [nel_data.drop(["nel_results"], axis=1), nel_data["nel_results"].apply(pd.Series)],
                            axis=1
                            )

                    # cache to disk
                    nel_data.to_csv(self.results_files[col], index = None)
                
                except FileNotFoundError:
                    log.info(f"File {self.input_data_files[col]} with extracted named entities not found. Entities must first be extracted before performing entity linking.")

    def _add_special_tokens_text(self, example):
        text = example[self.col2link]
        entity_start = int(example["start"])
        entity_end = int(example["end"])
        example[self.col2link] = text[:entity_start] + " [START] " + text[entity_start:entity_end] + " [END] " + text[entity_end:]

        return example

    def _add_special_tokens_single_entities(self, example):
        example[self.col2link] = "[START] " + example[self.col2link] + " [END]"
        return example

    def _batchify(self, iterable: Iterable[Any]) -> Iterable[Any]:
        length = len(iterable)
        for idx in range(0, length, self.batch_size):
            yield iterable[idx:min(idx + self.batch_size, length)]
        
