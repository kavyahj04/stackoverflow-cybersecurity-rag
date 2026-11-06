from bs4 import BeautifulSoup


def clean_body(raw_body):
    if not raw_body:
        return ""
    soup = BeautifulSoup(raw_body, "html.parser")
    return soup.get_text(separator=" ", strip=True)
