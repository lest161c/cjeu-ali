import json
import os

html_files = os.listdir('HTMLs')
celex_ids = json.load(open('celex_ids.json'))

print("number of files downloaded: ", len(html_files))
print("number of files to download: ", len(celex_ids))
print(len(html_files)/len(celex_ids)*100, "% of files downloaded")