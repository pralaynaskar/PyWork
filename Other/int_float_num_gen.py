import random

def generate_random_numbers_to_file(filename="ran_nums.txt", count=100000):
    """
    Generates a specified count of random numbers (a mix of integers and floats)
    and writes each number on a new line in a text file.

    Args:
        filename (str): The name of the file to write the numbers to.
        count (int): The total number of random numbers to generate.
    """
    try:
        with open(filename, 'w') as f:
            for _ in range(count):
                # Randomly decide whether to generate an integer or a float
                # 0 for integer, 1 for float (50/50 chance)
                choice = random.randint(0, 1)

                if choice == 0:
                    # Generate a random integer between -1000 and 1000
                    random_number = random.randint(-1000, 1000)
                else:
                    # Generate a random float between -1000.0 and 1000.0
                    random_number = random.uniform(-1000.0, 1000.0)

                f.write(str(random_number) + '\n')
        print(f"Successfully generated {count} mixed random numbers to '{filename}'")
    except IOError as e:
        print(f"Error writing to file '{filename}': {e}")

# Call the function to generate the numbers
generate_random_numbers_to_file()

