# Define the input and output file paths
input_file = "./data/music_valid.txt"
output_file = input_file

# Read the input file and process the data
with open(input_file, "r") as f:
    lines = f.readlines()

# Replace class names with "all" in each line
updated_lines = ["all" + line[line.find("/"):].strip() for line in lines]

# Write the updated data to the output file
with open(output_file, "w") as f:
    f.write("\n".join(updated_lines))

print("Class names have been replaced with 'all' and saved to", output_file)
