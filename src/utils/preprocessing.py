import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)
nltk.download('punkt', quiet = True)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> str:
        """
        Transforms raw text into clean list of mathematical tokens.
        """
        # 1. Lowercasing: Ensures 'Python' and 'python' map to the same vector.
        text = text.lower()

        # 2. Noise Removal:  Removing non-alphabetical chracters.
        # Numbers like '2023' are often noise unless specifically parsed.
        text = re.sub(r'[^a-z\s]', '', text)

        # 3. Tokenization: Splitting the continuum of text into descrete units(atoms).
        tokens = nltk.word_tokenize(text)

        # 4. Stopword Removal:  MAthematically, words like 'is' or 'this' have
        # high frequency but low information entropy. Removing them
        # improves the signal for our similarity models
        tokens = [t for t in tokens if t not in self.stop_words]

        # 5. Lemmatization: Reducing words to their dictionary base (lemma)
        # This maps 'running' and 'ran' to 'run', effectivaly grouping 
        # related points in out vector space.
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens
    



    

        