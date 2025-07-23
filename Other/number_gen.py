import random

def generate_random_numbers_to_file(filename="random_numbers.txt", count=1000): # Changed count to 1,000,000
    """
    Generates a specified count of random integers (positive, negative, and zero)
    and writes each number on a new line in a text file.

    Args:
        filename (str): The name of the file to write the numbers to.
        count (int): The number of random integers to generate.
    """
    try:
        with open(filename, 'w') as f:
            for _ in range(count):
                # Generate a random integer between -1000 and 1000 (inclusive)
                # You can adjust this range as needed.
                random_number = random.randint(-1000, 1000)
                f.write(str(random_number) + '\n')
        print(f"Successfully generated {count} random numbers to '{filename}'")
    except IOError as e:
        print(f"Error writing to file '{filename}': {e}")

# Call the function to generate the numbers
generate_random_numbers_to_file()

