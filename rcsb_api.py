import requests
import random
import os

# Step 1: Fetch All PDB IDs with Pagination
def fetch_all_pdb_ids():
    pdb_ids = []
    start = 0
    rows = 10000  # Fetch 10,000 PDB IDs per request
    count = 0

    while True:
        query = {
            "query": {
                "type": "terminal",
                "service": "text"
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": start,
                    "rows": rows
                }
            }
        }

        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        response = requests.post(url, json=query)
        data = response.json()
        # print(data)

        if response.status_code != 200 or 'result_set' not in data:
            raise Exception(f"Error in API response: {data.get('message', 'Unknown error')}")

        batch_ids = [entry['identifier'] for entry in data['result_set']]
        pdb_ids.extend(batch_ids)

        # If fewer results are returned than requested, we've reached the end
        count += 1
        if count > 10:
            break

        # Move to the next batch
        start += rows

    return pdb_ids

# # Step 2: Randomly Select 20000 PDB IDs
def select_random_pdb_ids(pdb_ids, number):
    return random.sample(pdb_ids[:], number)

# Step 3: Download FASTA and PDB Files for Selected PDB IDs
def download_fasta_and_pdb_files(pdb_ids, output_dir='content/protein_files'):
    fasta_dir = os.path.join(output_dir, 'fasta')
    pdb_dir = os.path.join(output_dir, 'pdb')
    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(pdb_dir, exist_ok=True)

    for pdb_id in pdb_ids:
        # Download FASTA file
        fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/download"
        fasta_output_path = os.path.join(fasta_dir, f"{pdb_id}.fasta")

        response_fasta = requests.get(fasta_url)
        # print(response_fasta)
        if response_fasta.status_code == 200:
            with open(fasta_output_path, 'w') as f:
                f.write(response_fasta.text)
            print(f"Downloaded FASTA: {pdb_id}.fasta")
        else:
            print(f"Failed to download FASTA: {pdb_id}.fasta")

        # Download PDB file
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        pdb_output_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

        response_pdb = requests.get(pdb_url)
        if response_pdb.status_code == 200:
            with open(pdb_output_path, 'w') as f:
                f.write(response_pdb.text)
            print(f"Downloaded PDB: {pdb_id}.pdb")
        else:
            print(f"Failed to download PDB: {pdb_id}.pdb")
        # break

# Main execution
if __name__ == "__main__":
    print("Fetching all PDB IDs...")
    all_pdb_ids = fetch_all_pdb_ids()

    print(f"Total PDB IDs fetched: {len(all_pdb_ids)}")

    print("Selecting 20000 random PDB IDs...")
    selected_pdb_ids = select_random_pdb_ids(all_pdb_ids, 20000)

    print("Downloading selected FASTA and PDB files...")
    download_fasta_and_pdb_files(selected_pdb_ids)

    print("Download completed.")
