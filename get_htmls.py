from eurlex import get_html_by_celex_id
import json
from tqdm import tqdm

# Retrieve and parse the document with CELEX ID "32019R0947" into a Pandas DataFrame
celex_ids = json.load(open('celex_ids.json'))
for id in tqdm(range(len(celex_ids))):
    celex_id = celex_ids[id]
    html = get_html_by_celex_id(celex_id)

    # Save html in HTMLs folder
    with open(f"HTMLs/{celex_id}.html", "w") as file:
        file.write(html)