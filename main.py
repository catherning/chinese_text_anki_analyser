import jieba
from collections import Counter
from typing import Union, List, Dict
import pandas as pd
import string
# import opencc
import chinese_converter
import re

def load_vocab(vocab: Union[str, int], hsk_dict: Dict[str, Dict]) -> set:
    """Load known vocabulary from CSV or HSK level."""
    if isinstance(vocab, int):  # vocab is an HSK level
        known_vocab = {word for word, info in hsk_dict.items() if info["level"] <= vocab}
    elif isinstance(vocab, str):  # vocab is a CSV file path
        import pandas as pd
        df = pd.read_csv(vocab)
        known_vocab = set(df.iloc[:, 0])  # assume first column contains words
    else:
        raise ValueError("vocab must be an int (HSK level) or str (path to CSV)")
    return known_vocab

# Chinese + Western punctuation set
chinese_punct = "，。！？；：「」、『』（）【】《》〈〉——……“”‘’·"
all_punct = set(string.punctuation + chinese_punct)

def preprocess_text(text: str,hsk_dict) -> list:
        
    latin_re = re.compile(r'[A-Za-z0-9]')
    text= chinese_converter.to_simplified(text)

    tokens = []
    for word in jieba.cut(text):
        word = word.strip()
        if not word or word in all_punct or latin_re.search(word):
            continue
        # si le mot complet est connu ou non chinois, garde tel quel
        if word in hsk_dict or not all('\u4e00' <= char <= '\u9fff' for char in word):
            tokens.append(word)
        else:
            # sinon découpe caractère par caractère
            tokens.extend(list(word))
    return tokens

def get_unknown_words(vocab: Union[str, int], words: list, hsk_dict: Dict[str, Dict]) -> tuple[list[str], float]:
    known_vocab = load_vocab(vocab, hsk_dict)
    total = len(words)
    
    unknown = [word for word in words if word not in known_vocab]
    known_count = total - len(unknown)
    
    coverage = round(known_count / total, 2) if total > 0 else 0.0
    return unknown, coverage

def estimate_hsk_level(words: list, hsk_dict: Dict[str, Dict]) -> int:
    level_counts = Counter()

    total_words = len(set(words))
    for word in words:
        if word in hsk_dict:
            level_counts[hsk_dict[word]['level']] += 1
        else:
            level_counts[7] +=1


    cumulative = 0
    for level in range(1, 8):
        cumulative += level_counts.get(level, 0)
        if cumulative / total_words >= 0.8:
            return level
    return 7


def get_unknowns_definition(unknown_words: List[str], hsk_dict: Dict[str, Dict]) -> List[Dict[str, str]]:
    output = []
    for word in set(unknown_words):
        definition = hsk_dict.get(word, {}).get("definition", "N/A")
        level = hsk_dict.get(word, {}).get("level", "N/A")
        output.append({"word": word, "level": level, "definition": definition})
    return sorted(output, key=lambda x: x['level'] if isinstance(x['level'], int) else 99)


def build_hsk_dict_from_csv(path: str) -> dict:
    df = pd.read_csv(path)
    hsk_dict = {}
    for _, row in df.iterrows():
        word = row['word']
        hsk_dict[word] = {
            "level": int(row['level']),
            "definition": row['definition']
        }
    return hsk_dict

def test_pipeline_with_sample_text():
    hsk_path = "data/hsk_vocabulary.csv"
    hsk_dict = build_hsk_dict_from_csv(hsk_path)
    
    sample_text = "我是学生，老师安排我们研究问题并解决。我是法国人，我会说法语。"
    vocab = 2  # Simulate known vocab as HSK level 2

    # Expected unknown words (based on simulated jieba output)
    expected_unknown = {'安排', '研究', '解决'}
    
    words = preprocess_text(sample_text,hsk_dict)

    unknowns,coverage = get_unknown_words(vocab, words, hsk_dict)
    hsk_level = estimate_hsk_level(words, hsk_dict)
    output = get_unknowns_definition(list(unknowns), hsk_dict)

    assert hsk_level == 4, "Expected HSK 7 words in the distribution"
    # assert unknowns == expected_unknown, f"Expected {expected_unknown}, got {unknowns}"
    # assert any(w['definition'] != "N/A" for w in output), "Definitions missing from output"

    print("✅ Test passed: pipeline handles sample text as expected.")

def main(text, vocab):
    """Main function to process text and vocabulary."""
    hsk_path = "data/hsk_vocabulary.csv"
    hsk_dict = build_hsk_dict_from_csv(hsk_path)

    words = preprocess_text(text,hsk_dict)
    
    unknown_words, coverage = get_unknown_words(vocab, words, hsk_dict)
    hsk_level = estimate_hsk_level(words, hsk_dict)
    output = get_unknowns_definition(unknown_words, hsk_dict)

    print(f"Unknown words: {unknown_words}")
    print(f"Coverage: {coverage}%")
    print(f"HSK level estimation: {hsk_level}")
    print("Output:", output)

# Run the test

# test_pipeline_with_sample_text()

if __name__ == "__main__":
    
    main(
        text="""
            小狗坐在地上""",
        vocab=5
    )