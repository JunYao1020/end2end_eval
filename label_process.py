import re
import Levenshtein


def decode(res):
    """
    去除坐标，置信度等信息

    :param res: 原始ocr结果
    :return: 仅包含文字信息的ocr结果
    """
    raw_res = [line[1][0] for line in res]
    return process_raw(raw_res)


def is_chinese(c):
    """
    判断是否是汉字

    :param c: 待判断的字符
    :return: 汉字返回true
    """
    return '\u4e00' <= c <= '\u9fff' and c != '年' and c != '月'


def process_real_dict(real):
    """
    处理真实标签转化成的字典
    :param real: 内容是真实标签的字典
    :return:
    """
    for key, value in real.items():
        value = process_raw(value)
        real[key] = value


def process_raw(raw):
    """
    分隔字符串中的中英文

    :param raw: 待处理的字符串集合
    :return: 处理完的字符串集合
    """
    res = []
    for r in raw:
        start = 0
        for i, c in enumerate(r):
            if i == 0:
                continue
            if is_chinese(c) and not is_chinese(r[i - 1]) \
                    or not is_chinese(c) and is_chinese(r[i - 1]):
                res.append(r[start: i])
                start = i
            if i == len(r) - 1:
                res.append(r[start:])
    return filter_processed_res(res)


def separate_eng_ch(raw_str):
    res = []
    start = 0
    for i, c in enumerate(raw_str):
        if i == 0:
            continue
        if is_chinese(c) and not is_chinese(raw_str[i - 1]):
            res.append(raw_str[start: i])
            start = i
        if i == len(raw_str) - 1:
            res.append(raw_str[start:])

    return res


def separate_ch_eng_single(raw_str):
    for i, c in enumerate(raw_str):
        if not is_chinese(c):
            return raw_str[: i], raw_str[i:].strip()
    return raw_str, ''


def is_vin_then_get_vin(raw_str):
    if (raw_str[:3].lower() == 'vin' or raw_str[:3].lower() == 'v1n') and len(raw_str) > 17:
        return raw_str[3:]
    if 15 < len(raw_str) < 20 and bool(re.search(r'\d', raw_str)):
        return raw_str
    return ''


def filter_processed_res(processed_res):
    """
    去除字符串列表中的特殊字符

    :param processed_res: 待处理字符串列表
    :return: 不包含只有特殊字符的字符串的列表
    """
    res = [re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", '', s) for s in processed_res]
    return [s for s in res if len(s) != 0]


def connect(beta, alpha):
    """
    将两个list中每对最相似的字符串组合在一起，放入只有两个元素的列表中，最后把所有这些组合的列表存入结果列表

    :param alpha: 第一个list
    :param beta: 第二个list
    :return: 包含所有的组合列表
    """
    result = []
    for a in alpha:
        best_score = 0
        best_b = ' '
        for b in beta:
            score = Levenshtein.ratio(a.lower(), b.lower())
            if score > best_score:
                best_score = score
                best_b = b
        pair = [a, best_b]
        result.append(pair)
    return result
