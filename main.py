import jieba
from collections import Counter
from typing import Union, List, Dict
import pandas as pd
import string
import chinese_converter
import re
from davia import Davia, run_server

app = Davia()

def load_vocab(vocab: Union[str, int], hsk_dict: Dict[str, Dict], additional_vocab=None, sep=",",position_word=2) -> set:
    """Load known vocabulary from CSV or HSK level."""
    known_vocab = {word for word, info in hsk_dict.items() if info["level"] <= vocab}
    if  additional_vocab:
        df = pd.read_csv(additional_vocab,sep=sep, encoding='utf-8')
        known_vocab = known_vocab.union(df.iloc[:, position_word-1])  # assume second column contains words
        
    # Add the single characters of each word, if count >= 2 
    counter_char = Counter([char for word in known_vocab if type(word)==str for char in word])
    counter_char = {char for char,count in counter_char.items() if count>=2}
    known_vocab = known_vocab.union(counter_char)
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
        tokens.append(word)
    return tokens

def get_unknown_words(text: list, known_vocab) -> tuple[list[str], float]:

    total = len(text)
    
    # Known if word in known_vocab, or all its characters are in known vocab if the word itself is not in it
    unknown = [word for word in text if word not in known_vocab and not set(word).issubset(known_vocab)]
    known_count = total - len(unknown)
    
    coverage = round(known_count / total, 2) if total > 0 else 0.0
    return set(unknown), coverage

def estimate_hsk_level(words: list, hsk_dict: Dict[str, Dict]) -> int:
    level_counts = Counter()

    total_words = len(set(words))
    for word in words:
        if word in hsk_dict:
            level_counts[hsk_dict[word]['level']] += 1
        else:
            level_counts[7] +=1

    # TODO: also return the repartition, if the person wants to see in details?
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
    # TODO: estimate HSK level of any word, as vocab is not exhaustive 
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
    expected_unknown = ['安排', '研究', '并', '解决', '法', '国', '法', '语']
    
    res = main(sample_text, vocab = vocab)

    assert res["hsk_level"] == 3, "Expected HSK 7 words in the distribution"
    assert res["coverage"] == 0.6, "Expected coverage of 0.6 with HSK level 2"
    assert list(res["unknowns_def"].keys()) == expected_unknown, f"Expected {expected_unknown}, got {list(res['unknowns_def'].keys())}"
    assert any(w['definition'] != "N/A" for w in res["unknowns_def"]), "Definitions missing from output"

    print("✅ Test passed: pipeline handles sample text as expected.")

@app.task
def main(text:str, vocab:int, additional_vocab:str=None, sep:str=",",position_word=2):
    """Pipeline to process Chinese text and analyze vocabulary coverage by user HSK level

    Args:
        text (str): Text to analyze
        vocab (int): HSK level of known vocabulary
        additional_vocab (str, optional): path to csv file containing the known vocabulary of the user, in addition to the HSK level. Defaults to None.
        sep (str, optional): separator of the csv file. Defaults to ",".

    Returns:
        coverage (float): Coverage of known words in the text in percentage 
        hsk_level : Estimated HSK level of the text
        output : List of unknown words with their definition and HSK level
    """
    # Get full known vocab
    hsk_path = "data/hsk_vocabulary.csv"
    hsk_dict = build_hsk_dict_from_csv(hsk_path)
    known_vocab = load_vocab(vocab, hsk_dict, additional_vocab, sep,position_word)

    text_processed = preprocess_text(text,hsk_dict)
    
    # Analyze text
    hsk_level = estimate_hsk_level(text_processed, hsk_dict)    
    unknown_words, coverage = get_unknown_words(text_processed, known_vocab)
    unknowns_def = get_unknowns_definition(unknown_words, hsk_dict)

    print(f"Coverage: {coverage}%")
    print(f"HSK level estimation: {hsk_level}")
    print(f"Unknown words : {len(unknowns_def)} || {unknown_words}")
    return {
        "coverage": coverage,
        "hsk_level": hsk_level,
        "unknowns_def": unknowns_def
    }
    
# TODO: method to add to known vocab, and can download/save, for later


if __name__ == "__main__":
    # test_pipeline_with_sample_text()
    
    main(
        text="""
        健康饮食的小对话

小明: 小红，你知道吗？最近我开始注重健康饮食了。

小红: 真的吗？那你都吃些什么？

小明: 我尽量多吃蔬菜水果，还有一些健康的蛋白质，比如鱼和鸡肉。

小红: 听起来很不错啊！我也想吃得更健康，但不知道从哪里开始。

小明: 没关系，我们可以一起制定一个健康饮食计划。你可以从减少糖分和油脂开始，增加蔬菜和全谷类食品的摄入。

小红: 那听起来很有挑战性，但我愿意尝试。你有什么好的食谱推荐吗？

小明: 我可以分享一些简单又健康的食谱给你，比如水煮蔬菜和烤鸡胸肉。我们还可以一起去市场购买新鲜的食材。

小红: 太好了！我期待我们一起迈向更健康的生活方式
""",
        vocab=1,
        # additional_vocab="data/anki_chinese.csv",
        sep="\t"
    )

#             近日，美国明尼阿波利斯市亨内平县检察官办公室公布对京东CEO刘强东事件的调查结果，决定对刘强东不予起诉，这意味着该案正式结案，刘强东无罪。
# 事件起因是在美国一个饭局过后，刘强东与女受害人在她的公寓发生性关系，随后，女方向警方报警，称遭到强奸，随后事件在中国社交媒体上发酵。但近日，美国律师放出消息，美警方已经宣布刘强东无罪。
# 刘强东在中国社交媒体上也做出道歉，称在女受害者房间所发生的事情都是男女自愿行为，虽不构成犯罪，但也对家庭造成了莫大的伤害，将会尽全力对家庭妻子孩子做出弥补。
    run_server(app)