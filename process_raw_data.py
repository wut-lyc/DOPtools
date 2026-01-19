import re
import csv
import os

def process_raw_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    
    # Pattern to identify the start of a new record (ID format: XXX_###)
    id_pattern = re.compile(r'^[A-Z0-9]+_\d+\s')

    # Skip header line
    start_idx = 0
    if lines[0].startswith("ID logTg"):
        start_idx = 1
    
    data_rows = []
    
    current_lines = []
    
    for i in range(start_idx, len(lines)):
        line = lines[i]
        if id_pattern.match(line):
            if current_lines:
                # Process previous
                full_line = "".join([l.strip('\n') for l in current_lines])
                data_rows.append(parse_line(full_line))
            current_lines = [line]
        else:
            current_lines.append(line)
            
    # Process last
    if current_lines:
        full_line = "".join([l.strip('\n') for l in current_lines])
        data_rows.append(parse_line(full_line))

    # Write to CSV
    header = ["ID", "logTg", "Serie", "Ref", "name", "SMILES", "SMI_generator"]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)
    
    print(f"Processed {len(data_rows)} records.")
    print(f"Saved to {output_file}")
    
    # Validation
    for i, row in enumerate(data_rows):
        if len(row) != 7:
            print(f"Warning: Row {i+1} (ID: {row[0]}) has {len(row)} columns.")

def parse_line(full_line):
    # Split by whitespace
    parts = full_line.split()
    
    p_id = parts[0]
    logTg = parts[1]
    serie = parts[2]
    ref = parts[3]
    
    smi_generator = parts[-1]
    
    # The middle part contains Name and SMILES (which might be fragmented)
    middle_parts = parts[4:-1]
    
    # Heuristic to separate Name from SMILES
    # We scan from left to right.
    # Name tokens usually contain characters NOT found in SMILES (e.g. 'a', 'e', 'm', 'y' etc not in element symbols).
    # SMILES tokens usually strictly follow a limited charset.
    
    # Allowed SMILES single characters (including aromatic). 
    # Note: 'l' is allowed only in 'Cl', 'Al'. 'r' only in 'Br'. 'i' in 'Si', 'Li'.
    # To simplify, we define a set of "Suspicious for SMILES" characters (Forbidden in standard SMILES).
    # Common name letters: a, d, e, g, j, k, m, q, r (except Br), t, u, v, w, x, y, z.
    # Also 'L', 'M', 'R', 'T', 'X', 'Y', 'Z', 'A', 'E', 'G', 'J', 'Q' are usually not single-char elements in organic chem (except maybe rare ones).
    # We will treat any token containing these as Name.
    # Also, standalone digits are treated as Name (SMILES won't be just '2').
    
    forbidden_chars = set("adegjkmqtuwxyzADEGJKMQRTUWXYZ")
    # 'r' is forbidden, but 'Br' contains 'r'. 'l' is forbidden, 'Cl' contains 'l'. 'i' forbidden, 'Si' contains 'i'.
    # We need to be careful checking "contains forbidden".
    # Better approach: Check if token consists ONLY of Allowed "SMILES Atoms/Symbols".
    
    # Allowed: C, c, H, h, O, o, N, n, P, p, S, s, F, f, I, B, b.
    # Multi-char: Cl, Br, Si, Li, Na, Mg, Al, K, Ca, Ba.
    # Non-alpha: 0-9, (, ), [, ], =, #, -, +, ., /, \, @, %.
    
    split_index = 0
    for i, token in enumerate(middle_parts):
        if is_likely_name_part(token):
            split_index = i + 1
            
    name_parts = middle_parts[:split_index]
    smiles_parts = middle_parts[split_index:]
    
    name = " ".join(name_parts)
    smiles = "".join(smiles_parts) # Join SMILES fragments without space
    
    return [p_id, logTg, serie, ref, name, smiles, smi_generator]

def is_likely_name_part(token):
    # 1. Standalone numbers are likely Name parts (e.g. "Perfluoropolymer 2")
    if token.isdigit():
        return True
        
    # 2. Check for forbidden characters that indicate a word
    # Remove valid multi-char elements from the token to avoid false positives
    # e.g. "Bromine" -> remove "Br" -> "omine" -> contains 'm', 'e' -> Name.
    # "Cl" -> remove "Cl" -> "" -> SMILES.
    
    temp_token = token
    # Common elements in this dataset
    for elem in ["Cl", "Br", "Si", "Li", "Na", "Mg", "Al", "Ba", "Ca"]:
        temp_token = temp_token.replace(elem, "")
        
    # Now check remaining string for forbidden chars
    # forbidden = "ade..."?
    # Actually, simpler: Does it contain any letter that is NOT in [c, h, o, n, p, s, f, i, b] (case insensitive)?
    # Wait, 'H' is allowed. 'h' is allowed (in []).
    # 'I' is allowed. 'i' is allowed (aromatic?).
    
    # Let's count "Name-like" chars.
    # Name-like: a, d, e, g, j, k, m, u, r, l, t, v, y, z...
    # (assuming we stripped Cl, Br, Si, Al etc).
    
    suspicious_chars = "adegjkmqrtuvwyz" + "adegjkmqrtuvwyz".upper()
    # Note: 'l' and 'L' are suspicious if not in Cl/Al.
    suspicious_chars += "lL"
    # 'x' and 'X' (halogen placeholder?) or name. 'X' often in names.
    suspicious_chars += "xX"

    for char in suspicious_chars:
        if char in temp_token:
            return True
            
    return False


if __name__ == "__main__":
    input_path = r"d:\Desktop\深度学习\描述符计算\DOPtools\raw_information"
    output_path = r"d:\Desktop\深度学习\描述符计算\DOPtools\processed_data.csv"
    process_raw_data(input_path, output_path)

