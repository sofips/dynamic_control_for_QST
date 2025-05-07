import pickle
import sys

def read_and_display_pkl(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("Contents of the .pkl file:")
            print(data)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pickle.UnpicklingError:
        print("Error: The file is not a valid pickle file.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = sys.argv[1]  # Replace with your .pkl file path
read_and_display_pkl(file_path)
