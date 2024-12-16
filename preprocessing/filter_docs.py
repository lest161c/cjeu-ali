import os
import argparse
import json
from bs4 import BeautifulSoup

def extract_sections(input_file):
    """
    Extract relevant sections (Summary, Grounds, Operative part) from the given HTML file.
    Returns the extracted content as a string.
    
    :param input_file: Path to the source HTML file
    :return: Extracted content as a string
    """
    # Read the input HTML file
    with open(input_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Define the headings we are interested in
    relevant_headings = ["Summary", "Grounds", "Operative part"]
    stop_headings = ["Questions", "Decision", "Answer", "Part", "Keywords", "Costs"]

    # Initialize a list to hold the extracted content
    extracted_content = []

    # Loop through the HTML to find relevant headings and their content
    current_section = None
    for element in soup.find_all(['h2', 'p']):
        # If we encounter a heading
        if element.name == 'h2' and element.get_text(strip=True) in relevant_headings:
            # Start a new section for relevant headings
            current_section = element.get_text(strip=True)
            extracted_content.append(f"\n\n{current_section}\n")
        elif element.name == 'h2' and element.get_text(strip=True) in stop_headings:
            # Stop extracting when we hit a stop heading
            current_section = None
        elif current_section:
            # Append the content of the paragraphs under relevant sections
            if element.name == 'p':
                extracted_content.append(element.get_text(strip=True))

    # Return the extracted content as a string
    return "\n".join(extracted_content)


def save_json(output_dir, extracted_docs):
    """
    Save all extracted documents into a single JSON file.
    
    :param output_dir: Directory where the JSON file will be saved
    :param extracted_docs: List of extracted content strings
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'extracted_documents.json')

    # Write the extracted documents to the JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(extracted_docs, json_file, ensure_ascii=False, indent=4)

    print(f"All documents extracted and saved to {output_file}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Extract relevant sections from HTML files')
    parser.add_argument('input_dir', type=str, help='Directory containing input HTML files')
    parser.add_argument('--output', type=str, default='filtered_doc', help='Directory to save the filtered documents')
    parser.add_argument('--json', action='store_true', help='Flag to save all extracted content into a single JSON file')

    # Parse arguments
    args = parser.parse_args()

    # List to store all extracted document contents
    all_extracted_docs = []

    # Get all HTML files in the input directory
    html_files = [f for f in os.listdir(args.input_dir) if f.endswith('.html')]

    # Process each HTML file
    for html_file in html_files:
        input_file = os.path.join(args.input_dir, html_file)
        
        # Extract sections from each HTML file
        extracted_content = extract_sections(input_file)

        # Only proceed if extracted content is not empty
        if extracted_content.strip():  # Check if the content is not just whitespace
            # Save to individual text files if needed
            if not args.json:
                output_file = os.path.join(args.output, f"extracted_{html_file.replace('.html', '.txt')}")
                with open(output_file, 'w', encoding='utf-8') as output:
                    output.write(extracted_content)
                print(f"Extraction complete. Output saved to {output_file}")
            
            # Add the extracted content to the list for JSON output
            all_extracted_docs.append(extracted_content)

    # If --json flag is set, save all extracted content to a single JSON file
    if args.json:
        save_json(args.output, all_extracted_docs)


if __name__ == "__main__":
    main()
