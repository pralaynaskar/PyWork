import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string
import os

class DataTokenizer:
    def __init__(self):
        """Initialize the tokenizer with necessary NLTK downloads."""
        self.download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
    
    def clean_text(self, text):
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits (optional)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_text(self, text, method='word', remove_stopwords=True, 
                     stem=False, lemmatize=False):
        """
        Tokenize text using various methods.
        
        Parameters:
        - text: Input text string
        - method: 'word' or 'sentence'
        - remove_stopwords: Boolean to remove stop words
        - stem: Boolean to apply stemming
        - lemmatize: Boolean to apply lemmatization
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        if method == 'sentence':
            return sent_tokenize(cleaned_text)
        
        # Word tokenization
        tokens = word_tokenize(cleaned_text)
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stop words
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming
        if stem:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def process_text_file(self, file_path, **kwargs):
        """Process a text file and return tokenized content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tokens = self.tokenize_text(content, **kwargs)
            return tokens
        
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"Error processing text file: {str(e)}")
            return None
    
    def process_csv_file(self, file_path, text_column=None, **kwargs):
        """Process a CSV file and return tokenized content."""
        try:
            df = pd.read_csv(file_path)
            
            # If no specific column is provided, show available columns
            if text_column is None:
                print("Available columns:")
                for i, col in enumerate(df.columns):
                    print(f"{i+1}. {col}")
                
                col_choice = input("Enter column number or name to tokenize: ")
                
                # Check if it's a number or column name
                try:
                    col_index = int(col_choice) - 1
                    if 0 <= col_index < len(df.columns):
                        text_column = df.columns[col_index]
                    else:
                        print("Invalid column number.")
                        return None
                except ValueError:
                    if col_choice in df.columns:
                        text_column = col_choice
                    else:
                        print("Invalid column name.")
                        return None
            
            # Process the selected column
            all_tokens = []
            for text in df[text_column].dropna():
                tokens = self.tokenize_text(str(text), **kwargs)
                all_tokens.extend(tokens)
            
            return all_tokens
        
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"Error processing CSV file: {str(e)}")
            return None
    
    def save_tokens(self, tokens, output_file):
        """Save tokens to a file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                if isinstance(tokens[0], list):  # For sentence tokenization
                    for i, sentence in enumerate(tokens):
                        file.write(f"Sentence {i+1}: {' '.join(sentence)}\n")
                else:  # For word tokenization
                    file.write('\n'.join(tokens))
            print(f"Tokens saved to '{output_file}'")
        except Exception as e:
            print(f"Error saving tokens: {str(e)}")

def main():
    """Main function to handle user interaction."""
    tokenizer = DataTokenizer()
    
    print("=== Data Tokenization Program ===")
    print("This program tokenizes text from TXT or CSV files.")
    
    while True:
        print("\nOptions:")
        print("1. Process Text File (.txt)")
        print("2. Process CSV File (.csv)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '3':
            print("Goodbye!")
            break
        
        if choice not in ['1', '2']:
            print("Invalid choice. Please try again.")
            continue
        
        # Get file path
        file_path = input("Enter the file path: ").strip()
        
        if not os.path.exists(file_path):
            print("File does not exist. Please check the path.")
            continue
        
        # Get tokenization options
        print("\nTokenization Options:")
        method = input("Tokenization method (word/sentence) [word]: ").strip().lower() or 'word'
        
        if method == 'word':
            remove_stopwords = input("Remove stop words? (y/n) [y]: ").strip().lower()
            remove_stopwords = remove_stopwords != 'n'
            
            stem = input("Apply stemming? (y/n) [n]: ").strip().lower() == 'y'
            lemmatize = input("Apply lemmatization? (y/n) [n]: ").strip().lower() == 'y'
        else:
            remove_stopwords = stem = lemmatize = False
        
        # Process the file
        print("\nProcessing file...")
        
        if choice == '1':  # Text file
            tokens = tokenizer.process_text_file(
                file_path, 
                method=method,
                remove_stopwords=remove_stopwords,
                stem=stem,
                lemmatize=lemmatize
            )
        else:  # CSV file
            tokens = tokenizer.process_csv_file(
                file_path,
                method=method,
                remove_stopwords=remove_stopwords,
                stem=stem,
                lemmatize=lemmatize
            )
        
        if tokens is None:
            continue
        
        # Display results
        print(f"\nTokenization complete! Found {len(tokens)} tokens.")
        
        # Show first 20 tokens as preview
        print("\nFirst 20 tokens:")
        for i, token in enumerate(tokens[:20]):
            print(f"{i+1}. {token}")
        
        if len(tokens) > 20:
            print("...")
        
        # Ask if user wants to save tokens
        save_choice = input("\nSave tokens to file? (y/n) [n]: ").strip().lower()
        if save_choice == 'y':
            output_file = input("Enter output filename (with .txt extension): ").strip()
            if not output_file.endswith('.txt'):
                output_file += '.txt'
            tokenizer.save_tokens(tokens, output_file)

if __name__ == "__main__":
    main()
