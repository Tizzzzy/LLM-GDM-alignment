import json
import os
from Bio import PDB


def pdb_to_json(pdb_folder, output_json_file):
    data = []

    if os.path.exists(output_json_file):
        with open(output_json_file, 'r') as json_file:
            data = json.load(json_file)

    existing_protein_ids = {protein["name"] for protein in data}
    count = 0

    for pdb_file in os.listdir(pdb_folder):
        if pdb_file.endswith(".pdb"):
            file_path = os.path.join(pdb_folder, pdb_file)
            protein_id = os.path.splitext(pdb_file)[0]

            if protein_id in existing_protein_ids:
                print(f"Skipping {protein_id}, already summarized.")
                continue

            try:
                parser = PDB.PDBParser()  
                structure = parser.get_structure('protein', file_path)
            except Exception as e:
                print(f"Skipping {protein_id}, could not parse file: {e}")
                continue

            seq = ""
            coords = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        coord = []
                        for atom in residue:
                            if atom.get_name() in ["N", "CA", "C", "O"]:
                                coord.append(atom.get_coord().tolist())
                        if len(coord) != 4:
                            continue
                        seq += residue.get_resname()[0]
                        coords.append(coord)

            if len(seq) != len(coords) or len(seq) == 0:
                continue
                
            protein_info = {
                "name": protein_id,
                "seq": seq,
                "coords": coords
            }

            data.append(protein_info)
            count += 1
            
            if count % 10 == 0:
                with open(output_json_file, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                print(f"Written {count} proteins to {output_json_file}")

                # break

    # Final write to ensure all data is saved
    with open(output_json_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Final write completed: {len(data)} proteins saved.")

    return data

json_result = pdb_to_json('../workspace/content2/protein_files/pdb', '../workspace_data/pdb_json.json')  

