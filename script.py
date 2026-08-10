import csv
import json
import logging
import os
import time
from typing import Any, Dict, List, Set
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

VOCAB_FILE = 'vocab.txt'
OUTPUT_CSV_FILE = 'anki_cards.csv'
CSV_HEADER = [
    'Word', 'Reading', 'POS', 'Meaning', 
    'Mnemonic', 'Example', 'Example_Reading', 'Example_VN'
]
MODEL_NAME = 'gemini-flash-latest'
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def load_vocab_words(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        logging.warning(f"File '{file_path}' does not exist.")
        return []
        
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def get_existing_words(csv_path: str) -> Set[str]:
    existing_words = set()
    if not os.path.exists(csv_path):
        return existing_words

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    existing_words.add(row[0].strip().lower())
    except Exception as e:
        logging.error(f"Error reading CSV file ({csv_path}): {e}")
        
    return existing_words


def build_gemini_prompt(words: List[str]) -> str:
    words_str = "\n".join(words)
    return f"""You are a Japanese linguistic expert. Please analyze the following list of Japanese words/phrases:
{words_str}

Return a JSON array where each object represents a word with the following schema:
- "word": original Japanese word
- "reading": Furigana / Reading / Romaji of the original word
- "pos": Part of speech
- "meaning": Concise Vietnamese translation of the original word
- "mnemonic": Detailed Sino-Vietnamese (Hán Việt) breakdown and Kanji meanings. (e.g. "ĐỒ (bản đồ) + THƯ (sách) + QUÁN (tòa nhà) = Tòa nhà lưu trữ sách vở, bản đồ")
- "example": Sample sentence in Japanese
- "example_reading": Furigana / Reading / Romaji of the example sentence
- "example_vn": Vietnamese translation of the example sentence"""


def parse_and_format_cards(raw_json: str) -> List[List[str]]:
    cards_data: List[Dict[str, Any]] = json.loads(raw_json)
    formatted_cards = []

    for item in cards_data:
        formatted_cards.append([
            item.get('word', ''),
            item.get('reading', ''),
            item.get('pos', ''),
            item.get('meaning', ''),
            item.get('mnemonic', ''),
            item.get('example', ''),
            item.get('example_reading', ''),
            item.get('example_vn', '')
        ])

    return formatted_cards


def append_to_csv(csv_path: str, new_cards: List[List[str]]) -> None:
    file_exists = os.path.exists(csv_path)
    is_empty = not file_exists or os.path.getsize(csv_path) == 0

    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if is_empty:
            writer.writerow(CSV_HEADER)
        writer.writerows(new_cards)


def generate_anki_cards_with_retry(client: genai.Client, prompt: str) -> List[List[str]]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            return parse_and_format_cards(response.text)

        except Exception as e:
            logging.error(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                logging.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.critical("Max retries exceeded. Aborting process.")
                raise e


def main() -> None:
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logging.error("Missing GEMINI_API_KEY environment variable.")
        exit(1)

    all_words = load_vocab_words(VOCAB_FILE)
    if not all_words:
        logging.info("No words found in input file.")
        return

    existing_words = get_existing_words(OUTPUT_CSV_FILE)
    words_to_process = [w for w in all_words if w.lower() not in existing_words]

    if not words_to_process:
        logging.info("All vocabulary words have already been processed.")
        return

    logging.info(f"Processing {len(words_to_process)} new vocabulary word(s)...")

    client = genai.Client(api_key=api_key)
    prompt = build_gemini_prompt(words_to_process)

    try:
        new_cards = generate_anki_cards_with_retry(client, prompt)
        if new_cards:
            append_to_csv(OUTPUT_CSV_FILE, new_cards)
            logging.info(f"Successfully saved {len(new_cards)} new card(s) to '{OUTPUT_CSV_FILE}'.")
    except Exception:
        logging.error("Failed to complete Anki card generation process.")
        exit(1)


if __name__ == '__main__':
    main()
