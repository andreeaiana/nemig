from typing import List
from pathlib import Path

import os
import re
from pandas.core.algorithms import duplicated
from tqdm import tqdm
import numpy as np
import pandas as pd
from langdetect import detect

from src.utils import pylogger
from src.dataset import utils

log = pylogger.get_pylogger(__name__)

# count decorator
def log_number_observations(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        log.info(f"Number of observations remaining: {result.shape[0]}")
        return result
    return wrapper

class NewsDataset():
    
    def __init__(
            self,
            language: str,
            overwrite_cache: bool,
            id_col: str,
            source_filename_col: str,
            news_outlet_col: str,
            provenance_col: str,
            author_col: str,
            author_org_col: str,
            creation_date_col: str,
            last_modified_col: str,
            keywords_col: str,
            content_col: str,
            title_col: str,
            description_col: str,
            body_col: str,
            formatted_content_col: str,
            language_col: str,
            article_length_col: str,
            number_subheaders_col: str,
            contains_forbidden_patterns_col: str,
            bool_drop_duplicates: bool, 
            bool_drop_foreign_articles: bool,
            bool_drop_outliers: bool,
            bool_drop_news_tickers: bool,
            bool_drop_articles_with_pattern: bool,
            pattern_list: List[str],
            author_sep_signs: List[str],
            subheader_threshold: int,
            max_words: int,
            scale_factor_outlier: float,
            raw_dataset_dir: str,
            data_raw_file: str,
            data_processed_file: str
            ) -> None:

        self.language = language
        self.overwrite_cache = overwrite_cache
            
        # Initial colums
        self.id_col = id_col
        self.source_filename_col = source_filename_col
        self.news_outlet_col = news_outlet_col
        self.provenance_col = provenance_col
        self.author_col = author_col
        self.author_org_col = author_org_col
        self.creation_date_col = creation_date_col
        self.last_modified_col = last_modified_col
        self.keywords_col = keywords_col
        self.content_col = content_col
        self.title_col = title_col
        self.description_col = description_col
        self.body_col = body_col
            
        # Pre-procesing columns
        self.formatted_content_col = formatted_content_col
        self.language_col = language_col
        self.article_length_col = article_length_col
        self.number_subheaders_col = number_subheaders_col
        self.contains_forbidden_patterns_col = contains_forbidden_patterns_col
        
        # Pre-processing variables
        self.bool_drop_duplicates = bool_drop_duplicates
        self.bool_drop_foreign_articles = bool_drop_foreign_articles
        self.bool_drop_outliers = bool_drop_outliers
        self.bool_drop_news_tickers = bool_drop_news_tickers
        self.bool_drop_articles_with_pattern = bool_drop_articles_with_pattern
            
        # Pre-processing constants
        self.pattern_list = pattern_list
        self.author_sep_signs = author_sep_signs
        self.subheader_threshold = subheader_threshold
        self.max_words = max_words
        self.scale_factor_outlier = scale_factor_outlier

        # Files and directories
        self.raw_dataset_dir = Path(raw_dataset_dir)
        self.data_raw_file = Path(data_raw_file)
        self.data_processed_file = Path(data_processed_file)

    def _load_news_files(self):
        articles_list = []
        id = 0
        
        for file in tqdm(list(self.raw_dataset_dir.rglob("*/json/*.json"))):
            try:
                # read file
                article = utils.read_json(file)

                if article is None:
                    continue

                # retrieving information from the json file
                url = article.get(self.provenance_col)
                news_outlet = article.get(self.news_outlet_col)
                creation_date = article.get(self.creation_date_col)
                last_modified = article.get(self.last_modified_col)
                content = article.get(self.content_col)
                title = content[self.title_col]
                description = content[self.description_col]
                body = content[self.body_col]
                keywords = article.get(self.keywords_col)
                author_person = article.get(self.author_col)
                author_org = article.get(self.author_org_col)
                
                # annotations based on content
                formatted_content = utils.format_content(title=title, body=body)
                language = detect(formatted_content)
                article_length = len(formatted_content.split(" "))
                number_subheaders = len(re.findall('<h2>', formatted_content))
                forbidden_pattern = utils.check_if_article_contains_forbidden_pattern(formatted_content, self.pattern_list)
                
                # summarize article in dict
                article_dict = {
                        self.source_filename_col: file,
                        self.id_col: id,
                        self.news_outlet_col: news_outlet,
                        self.provenance_col: url,
                        self.creation_date_col: creation_date,
                        self.last_modified_col: last_modified,
                        self.title_col: title,
                        self.description_col: description,
                        self.body_col: body,
                        self.formatted_content_col: formatted_content,
                        self.keywords_col: keywords,
                        self.author_col: author_person,
                        self.author_org_col: author_org,
                        self.language_col: language,
                        self.article_length_col: article_length,
                        self.number_subheaders_col: number_subheaders,
                        self.contains_forbidden_patterns_col: forbidden_pattern
                        }

                # append article
                articles_list.append(article_dict)
                
                # increment article id
                id += 1

            except Exception as e:
                log.info(f"Error at {file}")
                log.info(e)

        # store results in dataframe
        data = pd.DataFrame(articles_list)

        # create cache directory and subdirectories if not already created
        self.data_raw_file.parent.mkdir(exist_ok=True, parents=True)
        
        # export results as csv
        data.to_csv(self.data_raw_file, index=False)

        log.info(f"Raw dataset size: {len(data)} articles.")
        
        return data

    def _process_data(self, data = pd.DataFrame) -> pd.DataFrame:
        
        # drop duplicates
        if self.bool_drop_duplicates:
            log.info(f"Dropping duplicates.")
            data = self._drop_duplicates(data = data, duplicated_cols = [self.news_outlet_col, self.formatted_content_col])

        # drop all articles that are in a foreign language
        if self.bool_drop_foreign_articles:
            log.info(f"Dropping articles that are not in {self.language}.")
            data = self._drop_foreign_articles(data)

        # drop outliers (articles that are too long or short)
        if self.bool_drop_outliers:
            log.info(f"Dropping outlier, e.g. articles that are too long or too short.")
            data = self._drop_outliers(data)

        # drop news tickers
        if self.bool_drop_news_tickers:
            log.info(f"Dropping articles with more than {self.subheader_threshold} subheaders. Articles with more than"
                     f" {self.subheader_threshold} subheaders are considered news tickers.")
            data = self._drop_news_ticker(data)

        # drop articles with forbidden patterns
        if self.bool_drop_articles_with_pattern:
            log.info(f"Dropping artilces that contain a predefined regular expression. "
                     f"The predefined patterns are: {self.pattern_list}.")
            data = self._drop_articles_with_predefined_pattern(data)

        # reset indices
        data = data.reset_index(drop=True)

        # merge headlines and paragraphs
        data[self.body_col] = data[self.body_col].apply(lambda x: sum(x.values(), []))
        data[self.body_col] = data[self.body_col].apply(lambda x: " ".join([para for para in x if para != "" and para != " "]))

        # remove empty spaces from titles and descriptions
        data[self.title_col] = data[self.title_col].str.strip()
        data[self.description_col] = data[self.description_col].str.strip()

        # remove None values and fix formatting of values
        data[self.description_col] = data[self.description_col].fillna("")
        data[self.author_col] = data[self.author_col].apply(lambda authors: [author for author in authors if author is not None])
        data[self.author_col] = data[self.author_col].apply(lambda authors: authors[0] if len(authors) > 0 and type(authors[0]) == list else authors)
        data[self.author_org_col] = data[self.author_org_col].apply(lambda authors: [authors] if type(authors) == str else authors)

        # clean author names
        for sign in self.author_sep_signs:
            data[self.author_col] = data.apply(lambda row: self._clean_authors(row[self.author_col], sign), axis = 1)
            data[self.author_org_col] = data.apply(lambda row: self._clean_authors(row[self.author_org_col], sign), axis = 1)
        
        # export results as csv
        data.to_csv(self.data_processed_file, index=False)
        
        log.info(f"Preprocessed dataset size: {len(data)} articles.")
        
        return data

    def load_data(self) -> pd.DataFrame:
        if not self.overwrite_cache:
            try:
                # load cached processed data
                log.info(f"Loading cached processed {self.language} dataset.")
                data = pd.read_csv(self.data_processed_file)
                log.info(f"Dataset size: {len(data)} articles.")
            
            except FileNotFoundError:
                # data not processed yet; load and process raw data
                log.info(f"Cached processed dataset at {self.data_processed_file} not found.")
                try:
                    # load cached raw data
                    log.info(f"Loading the cached raw {self.language} dataset.")
                    data = pd.read_csv(self.data_raw_file)
                    log.info(f"Loaded raw dataset with {len(data)} articles.")
                    
                except FileNotFoundError:
                    # raw data not cached, read the individual data files
                    log.info(f"Cached raw dataset at {self.data_raw_file} not found. Loading the individual news files.")
                    data = self._load_news_files()

                # preprocess the raw dataset
                log.info(f"Preprocessing the raw dataset.")
                data = self._process_data(data)
                log.info(f"Dataset size: {len(data)} articles.")
        else:
            log.info(f"Loading the individual news files.")
            data = self._load_news_files()
            
            # preprocess the raw dataset
            log.info(f"Preprocessing the raw dataset.")
            data = self._process_data(data)
            log.info(f"Dataset size: {len(data)} articles.")
        
        return data

    @log_number_observations
    def _drop_duplicates(self, data: pd.DataFrame, duplicated_cols: List[str]) -> pd.DataFrame:
        duplicates = data[duplicated_cols].duplicated()
        data = data[duplicates == False]
        
        return data

    @log_number_observations
    def _drop_foreign_articles(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop(data[data[self.language_col] != self.language].index)
        
        return data

    @log_number_observations
    def _drop_outliers(self, data: pd.DataFrame) -> pd.DataFrame:

        # log transform article length distribution so that it resembles a normal distribution
        log_article_length = np.log(data[self.article_length_col])

        # get outlier bounds
        lower_bound, upper_bound = utils._get_outlier_bounds(series = log_article_length, scale_factor = self.scale_factor_outlier)

        # get indices of articles to drop
        is_outlier = (log_article_length.between(lower_bound, upper_bound) == False)
        outlier_index = data[is_outlier].index

        # drop outliers
        data = data.drop(outlier_index)

        return data

    @log_number_observations
    def _drop_news_ticker(self, data: pd.DataFrame) -> pd.DataFrame:
        idx2drop = data[data[self.number_subheaders_col] > self.subheader_threshold].index
        data = data.drop(idx2drop)

        return data

    @log_number_observations
    def _drop_articles_with_predefined_pattern(self, data = pd.DataFrame) -> pd.DataFrame:
        idx2drop = data[data[self.contains_forbidden_patterns_col] == True].index
        data = data.drop(idx2drop)

        return data

    @log_number_observations
    def drop_news_ticker(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Drops articles that are news or live ticker. 
        Assumption: article contains more than than a specified number of subheadings.

        Args:
            data (pd.Dataframe): Dataframe of news articles.
            threshold (int): Threshold for the number of subheadings contained in a news ticker.

        Returns:
            pd.Dataframe: The dataframe without news tickers.
        """
        idx2drop = data[data[self.number_subheaders_col] > self.subheader_threshold].index
        data = data.drop(idx2drop)

        return data

    def _clean_authors(self, authors_list: List[str], sign: str) -> List[str]:
        """
        Cleans a list of author names by splliting them based on a given sign.

        Args: 
            authors_list (List[str]): A list of author names.
            sign (str): Sign that separates author names in the list.

        Returns:
            List[str]: A list of splitted author names.
        """
        if authors_list:
            authors = list()
            for author in authors_list:
                if sign in author:
                    authors.extend([name.strip() for name in author.split(sign)])
                else:
                    authors.append(author)
            return authors
        
        return authors_list

