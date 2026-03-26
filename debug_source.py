"""See what the actual source of load_single_data_file is."""

import inspect

from paradigma.load import load_single_data_file

source = inspect.getsource(load_single_data_file)
print("First 1500 chars of load_single_data_file source:")
print(source[:1500])
print("\n...")
print("\nLast 500 chars of source:")
print(source[-500:])

print("\n\nFile location:")
print(inspect.getfile(load_single_data_file))
