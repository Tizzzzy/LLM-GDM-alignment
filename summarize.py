from openai import OpenAI
import json
import os

client = OpenAI(api_key="sk-")

def gpt_query(protein_id, fasta_text, temp):
    MODEL="gpt-4o"

    instruction_prompt = "You are a biologist with expertise in protein sequence analysis. Your task is to summarize complex protein sequence data into two or three sentences that highlight key features such as molecute type, chains, structural motifs, organism, etc."
    query = f"""Summarize the following protein knowledge, start with the sentence: '{temp}'\n\n{protein_id}:\n{fasta_text}"""

    completion = client.chat.completions.create(
      model=MODEL,
      messages=[
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": query}
      ]
    )
    print("Assistant: " + completion.choices[0].message.content)
    return completion.choices[0].message.content


def process_fasta_files(fasta_folder):
    protein_summaries = {}
    count = 0
    json_file_path = "protein_summaries.json"

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            protein_summaries = json.load(json_file)

    for fasta_file in os.listdir(fasta_folder):
        if fasta_file.endswith(".fasta"):
            file_path = os.path.join(fasta_folder, fasta_file)
            protein_id = os.path.splitext(fasta_file)[0]

            if protein_id in protein_summaries:
                print(f"Skipping {protein_id}, already summarized.")
                continue

            sequence_length = 0

            # Read the FASTA file content
            with open(file_path, 'r') as file:
                fasta_text = file.read()
                sequence_count = fasta_text.count('>')

                file.seek(0)

                for line in file:
                    line = line.strip()
                    if not line.startswith('>'):
                        sequence_length += len(line)
                # print(sequence_length)

            if sequence_count > 1:
                temp = f"""The protein structure {protein_id} has a sequence length of: {sequence_length} amino acids."""
                summary = gpt_query(protein_id, fasta_text, temp)

            else:
                try:
                    lines = fasta_text.splitlines()
                    first_line = lines[0].strip() if lines else ''
                    parts = first_line.split('|')
                    details = f"The protein structure {protein_id} involves the following chains: {parts[1]}. The protein is named {parts[2]}. It was derived from the organism {parts[3]}."
                    summary = f"""The protein structure {protein_id} has a sequence length of: {sequence_length} amino acids. Here is more information: {details}"""
                except Exception as e:
                    # Log or handle the exception as needed
                    print(f"An error occurred: {e}")
                    continue

            # Store the protein ID and summary in the dictionary
            protein_summaries[protein_id] = summary
            count += 1

            if count % 10 == 0:
                with open(json_file_path, "w") as json_file:
                    json.dump(protein_summaries, json_file, indent=4)
                print(f"Updated {json_file_path} with {count} summaries.")

                # break

    # Save the summaries to a JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(protein_summaries, json_file, indent=4)
    print(f"Final update to {json_file_path} with {count} summaries.")


fasta_folder = "content/protein_files/fasta"
process_fasta_files(fasta_folder)
