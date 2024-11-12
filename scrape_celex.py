import requests
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm  # Import tqdm for progress indication

#total_pages = 1282
total_pages = 3
celex_ids = []

# Define the pattern to match CELEX IDs
pattern = r'CELEX:(.*?)(?=[\'"]|$)'

# Use tqdm to create a progress bar for the loop
for page_number in tqdm(range(1, total_pages + 1), desc="Scraping pages"):
    # Construct the URL for the current page
    url = f"https://curia.europa.eu/juris/documents.jsf?nat=or&mat=or&pcs=Oor&jur=C%2CT&for=&jge=&dates=&language=en&pro=&cit=none%252CC%252CCJ%252CR%252C2008E%252C%252C%252C%252C%252C%252C%252C%252C%252C%252Ctrue%252Cfalse%252Cfalse&oqp=&td=%3B%3B%3BPUB1%3BNPUB1%3B%3B%3BORDALL&avg=&lgrec=en&page={page_number}&lg=&cid=5749683#"
    
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:

        html_content = response.content.decode('utf-8')  # Decode bytes to string
        print(html_content)
        
        # Find all text that matches the CELEX pattern
        matches = re.findall(pattern, html_content)
        
        # Add the found CELEX IDs to the list
        celex_ids.extend(matches)
    else:
        print(f"Failed to retrieve page {page_number}: {response.status_code}")

# Write the CELEX IDs to a JSON file
with open('celex_ids.json', 'w') as json_file:
    json.dump(celex_ids, json_file)

# Print the total number of unique CELEX IDs found
print(f"Total CELEX IDs found: {len(celex_ids)}")
