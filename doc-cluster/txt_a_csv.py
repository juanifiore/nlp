import json
import csv

# Input and output file paths
input_file = "datasets/topics_sent1.txt"  # Replace with your .txt file path
output_file = "topic_sentences.csv"  # Replace with your desired .csv file path

# Initialize a list to store rows for the CSV
data_rows = []

# Read the .txt file line by line
with open(input_file, "r") as txt_file:
    for line in txt_file:
        # Parse each line as JSON
        record = json.loads(line.strip())
        # Extract 'text' and 'label' and append as a row
        data_rows.append([record["text"], record["label"]])

# Write to a .csv file
with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    # Write the header
    writer.writerow(["text", "label"])
    # Write the data rows
    writer.writerows(data_rows)

print(f"Converted {input_file} to {output_file} successfully.")

