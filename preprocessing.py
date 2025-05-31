import re
import string
import json
import requests
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Pembersihan teks dasar
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # remove mentions
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # remove hashtag
    text = re.sub(r'RT[\s]', '', text)          # remove RT
    text = re.sub(r"http\S+", '', text)         # remove link
    text = re.sub(r'[0-9]+', '', text)          # remove numbers
    text = re.sub(r'[^\w\s]', '', text)         # remove non-alphanumeric
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Case folding
def casefoldingText(text):
    return text.lower()

# Tokenisasi
def tokenizingText(text):
    return word_tokenize(text)

# Ambil stopwords dari GitHub
def load_stopwords_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        stopwords_list = response.text.splitlines()
        return set(stopwords_list)
    else:
        raise Exception("Gagal fetch stopwords: " + url)

# URL stopwords (ganti jika perlu)
stopwords_url = "https://raw.githubusercontent.com/andreass7/project-analisis-sentiment/refs/heads/master/kamus/stopwords/stopwords.txt"
custom_stopwords = load_stopwords_from_url(stopwords_url)
custom_stopwords.update([
    'iya', 'yaa', 'gak', 'nya', 'na', 'sih', 'ku', 'di', 'ga', 'ya', 'gaa', 'loh', 'kah', 'woi', 'woii', 'woy',
    'yg', 'efisiensi', 'anggaran', 'ri', 'kebijakan', 'prabowo', 'dpr', 'amp', 'rp', 'mah', 'indonesia', 'biar',
    'kena', 'bikin', 'wkwk', 'eh', 'min', 'efisiensinya', 'efisien', 'fex', 'aewe', 'p', 'mpa', 'swa', 'rakyat', 'pemerintah'
])

# Filtering stopwords
def filteringText(text):
    return [word for word in text if word not in custom_stopwords]

# Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmingText(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Gabungkan list token ke kalimat
def toSentence(list_words):
    return ' '.join(list_words)

# Ambil slangwords dari GitHub
slang_url = 'https://raw.githubusercontent.com/andreass7/project-analisis-sentiment/master/kamus/slangwords/slangword.txt'

def fetch_slangwords():
    response = requests.get(slang_url)
    if response.status_code == 200:
        text = response.text.strip()
        if not text.startswith('{'):
            text = '{' + text
        if not text.endswith('}'):
            text = text.rstrip(',') + '}'
        try:
            slang_dict = json.loads(text)
            return {k.lower(): v.lower() for k, v in slang_dict.items()}
        except json.JSONDecodeError as e:
            print("Gagal parsing slangwords:", e)
            return {}
    else:
        print(f"Gagal fetch slangwords. Status: {response.status_code}")
        return {}

# Inisialisasi slangwords
slangwords = fetch_slangwords()

# Ganti slangwords
def fix_slangwords(text):
    words = text.split()
    fixed_words = [slangwords.get(word.lower(), word) for word in words]
    return ' '.join(fixed_words)
