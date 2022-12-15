""" NLP methods mainly for the normalization of node names. """

import re
import validators
import unicodedata
from titlecase import titlecase


def _regularize_spaces(text: str) -> str:
    """ Replaces spaces in a string with underscores. """
    return text.replace(' ', '_')


def _capitalize(text: str) -> str:
    if not text:
        return text
    return titlecase(text)


def get_canonical_label(text: str) -> str:
    """ Returns the canonical label of a string by capitalizing and regularizing spaces. """ 
    if text != None:
        return _regularize_spaces(_capitalize(normalize_unicodedata(text)))
    return text


def _remove_unbalanced_bracket_content(text: str) -> str:
    """ Removes unbalanced brackets (i.e. unclosed/unopened) and their corresponding content. """
    open_brackets = tuple('([{')
    close_brackets = tuple(')]}')
    mapping = dict(zip(open_brackets, close_brackets))
    bracket_queue = []
    position_queue = []

    for pos, char in enumerate(text):
        if char in open_brackets:
            bracket_queue.append(mapping[char])
            position_queue.append(pos)
        elif char in close_brackets:
            if not bracket_queue or char != bracket_queue.pop():
                text = text[text.index(char)+1:]
                return _remove_unbalanced_bracket_content(text)

    if not bracket_queue:
        return text
    else:
        return text[:position_queue.pop()]


def _remove_trailing_chars(text: str) -> str:
    """ Removes trailing characters from the beginning or end of a string. """
    chars = ['.', '@', '/', '&', '-', "'"]
    for char in chars:
        text = text.strip(char)
    return text


def clean_label(text: str, entity_type: str) -> str:
    """ Cleans labels by removing brackets and certain characters. """
    
    # If the string is a valid domain, URL, or email address, return it without further processing
    if validators.domain(text) or validators.url(text) or validators.email(text):
        return text

    # Remove unicode characters
    text = unicodedata.normalize('NFKD', text)
    text = text.strip('\u200b')
    text = text.replace('\N{SOFT HYPHEN}', '')

    # Remove unbalanced bracket content
    text = _remove_unbalanced_bracket_content(text)

    # Remove trailing non-alphanumeric characters
    text = re.sub('\"|\“|\`|\´|\#|\«|\»|\>|\<|\,|\?|\!|\:|\;|\›|\‹|\„|\‚', ' ', text)
    text = re.sub('\*', '', text)
    text = re.sub('\s?\-+\s?|\s?\–+\s?', '-', text)
    text = re.sub('\.([a-zA-Z]{1})', '\\1', text)
    text = re.sub('\.\s([a-zA-Z]{1})\.', '\\1', text, count=2)
    text = re.sub('\.([a-zA-Z]{1})\.', '\\1', text, count=2)
    text = _remove_trailing_chars(text)

    # Remove numbers if entity type is PER (e.g. '2015.Erdogan' -> '.Erdogan')
    if entity_type == 'PER':
        text = re.sub('(\d\W?)', '', text)

    # Remove dots if entity type is PER or ORG (e.g. 'Matthew C. Perry' -> 'Matthew C Perry')
    if entity_type in ['PER', 'ORG']:
        text = re.sub('(\.)', ' ', text)

    # Remove @ and .domain if entity is PER or LOC
    if entity_type in ['PER', 'LOC']:
        text = re.sub('(\@\w*(\.\w*))', '', text)
        text = re.sub('(\.de|\.com|\.org|\.net)', '', text)

    # Lowercase, normalize spaces
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = text.lower()

    return text
    

def is_emoji(text: str) -> bool:
    """ Checks if a given string is an emoji. """
    emoji_pattern = re.compile(
        u'([\U0001F1E6-\U0001F1FF]{2})|' # flags
        u'([\U0001F600-\U0001F64F])'     # emoticons
        "+", flags=re.UNICODE)
    
    if not re.search(emoji_pattern, text):
        return False
    return True


def normalize_unicodedata(text: str) -> str:
    """ Applies unicodedata normalization of the form 'NFC'. """
    return unicodedata.normalize('NFC', text)
