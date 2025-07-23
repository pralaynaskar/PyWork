import random
import os

def generate_sentence():
    """Generates a grammatically simple, random sentence."""

    # Define word categories
    subjects = ["The cat", "A dog", "My friend", "The old man", "A bird", "The scientist", "The child", "The car"]
    verbs = ["ran", "jumped", "slept", "ate", "sang", "coded", "drove", "painted"]
    adjectives = ["quick", "happy", "blue", "loud", "tiny", "clever", "bright", "dark"]
    adverbs = ["quickly", "happily", "loudly", "slowly", "carefully", "eagerly", "silently", "smoothly"]
    objects = ["the ball", "a book", "some food", "the song", "a program", "the road", "a picture", "the tree"]
    prepositions = ["on", "under", "beside", "near", "through", "with", "around", "behind"]
    nouns = ["house", "park", "mountain", "city", "river", "forest", "sky", "ocean"]

    # Sentence structures (can be expanded for more variety)
    # Structure 1: Subject + Verb + Object.
    # Structure 2: Subject + Verb + Adverb.
    # Structure 3: Subject + Verb + Preposition + Noun.
    # Structure 4: Adjective + Subject + Verb + Object.
    sentence_structures = [
        lambda: f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}.",
        lambda: f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(adverbs)}.",
        lambda: f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(prepositions)} {random.choice(nouns)}.",
        lambda: f"{random.choice(adjectives).capitalize()} {random.choice(subjects).lower().replace('the ', '')} {random.choice(verbs)} {random.choice(objects)}."
    ]

    # Randomly choose a sentence structure and generate the sentence
    sentence = random.choice(sentence_structures)()

    # Capitalize the first letter and ensure it ends with a period
    sentence = sentence.strip()
    if not sentence.endswith('.'):
        sentence += '.'
    return sentence[0].upper() + sentence[1:]

def main():
    """Main function to get user input and generate sentences."""
    print("--- Random Sentence Generator ---")

    while True:
        try:
            num_sentences = int(input("Enter the number of sentences to generate (e.g., 1000, 50000): "))
            if num_sentences <= 0:
                print("Please enter a positive number.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    while True:
        output_filename = input("Enter the output filename (e.g., sentences.txt): ")
        if output_filename.strip():
            break
        else:
            print("Filename cannot be empty.")

    print(f"\nGenerating {num_sentences} sentences and saving to '{output_filename}'...")

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            for i in range(1, num_sentences + 1):
                sentence = generate_sentence()
                f.write(sentence + "\n")

                # Print progress for every 1000 sentences or 10% for smaller numbers
                if num_sentences > 1000 and i % 1000 == 0:
                    print(f"Generated {i}/{num_sentences} sentences...")
                elif num_sentences <= 1000 and i % (num_sentences // 10 or 1) == 0:
                    print(f"Generated {i}/{num_sentences} sentences...")


        print(f"\nSuccessfully generated {num_sentences} sentences and saved to '{output_filename}'.")
        print(f"File size: {os.path.getsize(output_filename) / (1024 * 1024):.2f} MB")

    except IOError as e:
        print(f"Error writing to file '{output_filename}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
