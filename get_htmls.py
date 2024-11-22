from eurlex import get_html_by_celex_id
import json
from tqdm import tqdm
import os

# List of CELEX IDs in HTMLs folder, used to avoid re-downloading the same documents
html_files = os.listdir('HTMLs')
html_celex_ids = [html_file.split('.')[0] for html_file in html_files]


celex_ids = json.load(open('celex_ids.json'))
for id in tqdm(range(len(celex_ids))):
    if celex_ids[id] in html_celex_ids:
        continue
    celex_id = celex_ids[id]
    html = get_html_by_celex_id(celex_id)

    # Save html in HTMLs folder
    with open(f"HTMLs/{celex_id}.html", "w") as file:
        file.write(html)