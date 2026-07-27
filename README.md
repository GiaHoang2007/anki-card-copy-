# anki-card-copy-
# Anki Card Generator (for Vietnamese)

An automated tool built with Python and Google Gemini API that automatically generates detailed flashcards for Anki from a simple vocabulary list.

## Features

- **Automated Card Generation:** Processes raw vocabulary from `vocab.txt` and generates structured flashcard data including reading, part of speech, meanings, mnemonics, and example sentences.
- **Duplicate Prevention:** Checks existing records in `anki_cards.csv` to avoid re-processing previously generated words.
- **Robust Error Handling:** Features built-in retry mechanisms for API calls and clear structured logging.
- **CI/CD Integration:** Configured with GitHub Actions to automatically process new vocabulary and keep the card database updated.

## Tech Stack

- **Language:** Python 3.x
- **AI Integration:** Google Gemini API (`google-genai` SDK)
- **Automation:** GitHub Actions

## Quick Start

### 1. Prerequisites

Set up your Gemini API key as an environment variable:
export GEMINI_API_KEY="your-api-key-here"
### 2. Installation

Clone the repository and install the required library:
git clone 
cd 
pip install google-genai
### 3. Usage

1. Add target vocabulary words to `vocab.txt` (one word per line).
2. Run the main script:
python script.py
3. Updated cards will be appended to `anki_cards.csv`, ready for import into Anki.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
