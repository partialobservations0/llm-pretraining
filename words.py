import urllib.request
import logging
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
import re

logger = logging.getLogger("Words")
logger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

def download_url(url, filepath):
    urllib.request.urlretrieve(url, filepath)
    logger.info("Downloaded to " + filepath)

def read_file(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        raw_text = f.read()
    logger.info("Length of text: " + str(len(raw_text)))
    return raw_text

def tokens_from_text(text):
    result = re.split(r'([,.:;?_!"()\']|--|[\s])', text)
    result = [item for item in result if item.strip()]
    logger.info("Tokens: " + str(len(result)))
    return result

def create_vocab(text):
    tokens = tokens_from_text(text)
    all_words = sorted(set(tokens))
    vocab = {w:i for i,w in enumerate(all_words)}
    
    logger.info("Vocab: " + str(len(vocab)))
    return vocab



#if __name__ == "__main__":
#    download_url(url, "the-verdict.txt")
#    text = read_file("the-verdict.txt")
#    create_vocab(text)

