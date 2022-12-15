from typing import List

import re
import json
import pandas as pd
from bs4 import BeautifulSoup

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def read_json(file):
    try:
        with open(file, "r") as json_data:
            data = json.loads(json_data.read())
        return data
    except Exception:
        log.info(f"Error reading {file}")
        return None


def check_if_article_contains_forbidden_pattern(article: str, pattern_list: List[str]) -> bool:
    """
    Checks if an article contains a predefined pattern.

    Args: 
        article (str): A news article.
        pattern_list (List[str]): A list of patterns.

    Returns:
        bool: Whether the article contains a predefined pattern.
    """
    for pattern in pattern_list:
        if re.findall(pattern, article):
            return True

    # No pattern was found in the article
    return False


def _add_html_tag(text: str, tag: str) -> str:
    return f"<{tag}>{text}</{tag}>"


def _get_paragraphs(paragraphs: List[str]) -> List[str]:
    paragraphs = [_add_html_tag(paragraph, "p") for paragraph in paragraphs if not re.findall("trends.embed.renderExploreWidget", paragraph)]
    return paragraphs


def _remove_links(text: str) -> str:
    """
    Regex based on https://gist.github.com/gruber/8891611

    Args:
        text (str): An input text.

    Returns:
        str: The text with the links removed.
    """
    text = re.sub(r'http\S+', '', text)

    return text


def _remove_html_tags(text: str) -> str:
    
    # prevent missing white space in further deletion of html symbols
    text = re.sub('<p>', ' ', text)

    # add punctuation after headings
    text = re.sub('</h1>', '. ', text)
    text = re.sub('</h2>', '. ', text)

    soup = BeautifulSoup(text, features="html.parser")
    text = soup.get_text()

    return text


def format_content(title: str, body: dict) -> str:
    text = _add_html_tag(title, "h1")

    for header, paragraphs in body.items():

        # avoid empty paragraphs
        if not paragraphs:
            continue

        # correct wrongly formatted header
        if len(header.split(" ")) > 50:
            header_html = _add_html_tag(header, "p") if header != "" else ""
        else:
            # reformat header
            header_html = _add_html_tag(header, "h2") if header != "" else ""

        # get paragraphs
        paragraphs = _get_paragraphs(paragraphs=paragraphs)

        # reformat paragraphs
        paragraphs_html = " ".join(paragraphs)

        text += " " + header_html + " " + paragraphs_html

    return text


def _get_outlier_bounds(series: pd.Series, scale_factor: float):
    
    # calculate statistics
    quantile_25 = series.quantile(0.25)
    quantile_75 = series.quantile(0.75)
    iqr = quantile_75 - quantile_25

    # calculate lower and upper bound
    lower_bound = quantile_25 - scale_factor * iqr
    upper_bound = quantile_75 + scale_factor * iqr

    return lower_bound, upper_bound
