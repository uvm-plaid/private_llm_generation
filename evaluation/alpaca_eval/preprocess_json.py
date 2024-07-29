import json

def fix_json(json_string):
    # This function aims to fix common JSON format issues, specifically escaping internal quotes
    # Escape internal double quotes that are not already escaped
    corrected_string = json_string.strip()
    corrected_string = corrected_string.replace('\\"', '"')  # Temporarily unescape to avoid double escaping
    corrected_string = corrected_string.replace('"', '\\"')  # Properly escape all double quotes
    return corrected_string

corrected_lines = []
with open('impossible_pubmed.json', 'r') as file:
    for line in file:
        if line.strip():  # Ignore empty lines
            corrected_line = fix_json(line)
            corrected_lines.append(corrected_line)

# Combine lines into a single JSON array
corrected_json = "[" + ",".join(corrected_lines) + "]"

try:
    data = json.loads(corrected_json)  # Attempt to parse the corrected JSON
    with open('corrected_impossible_pubmed.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)  # Save the valid JSON
except json.JSONDecodeError as e:
    print("Failed to decode JSON:", e)
    # Save the problematic JSON for inspection
    with open('debug_impossible_pubmed.json', 'w') as debug_file:
        debug_file.write(corrected_json)
    # Print around the error position for quick inspection
    error_char_index = e.pos  # Position of the error
    print("Error around: ...{}...".format(corrected_json[max(0, error_char_index-50):error_char_index+50]))
