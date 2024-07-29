import os
from pathlib import Path
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup  # Ensure BeautifulSoup is installed with pip install beautifulsoup4

def download_file_from_google_drive(file_id: str, destination: str) -> None:
    """Download a file from Google Drive, handling the confirmation for large files."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    # Make the initial request
    response = session.get(URL, params={'id': file_id}, stream=True)
    soup = BeautifulSoup(response.text, 'html.parser')
    token = get_confirm_token(soup)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(soup):
    """Extract the Google Drive download warning token from the HTML page."""
    tag = soup.find('a', id='uc-download-link')
    if tag:
        download_url = tag.get('href')
        token_value = download_url.split('confirm=')[1].split('&')[0]
        return token_value
    return None

def save_response_content(response, destination):
    """Save the content of a response to a destination file."""
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    Path("./data").mkdir(parents=True, exist_ok=True)
        # FILE_ID = "1-5kNodoxQWl5NOkR3VKliYndwz7U6tD3"
    # DESTINATION = "./data/dp_pubmed.json"
    FILE_ID ="1-941XYEQg5Z77bGl8KYw55eiUQL93IKi"
    DESTINATION = "./data/impossible_pubmed.json"


# https://drive.google.com/file/d/1-5kNodoxQWl5NOkR3VKliYndwz7U6tD3/view?usp=drive_link
# https://drive.google.com/file/d/1-941XYEQg5Z77bGl8KYw55eiUQL93IKi/view?usp=sharing