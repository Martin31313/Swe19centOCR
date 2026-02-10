# This is the repetition trimmer used in our thesis. We found that
# max_pattern_length=300 and threshold_chars=300 were good parameter
# values for a multitude of models on our validation set.

def repetition_trimmer(text, max_pattern_length=300, threshold_chars=300):
    if not text:
        return text, None, None
    
    text_len = len(text)
    true_max_L = min(max_pattern_length, text_len)

    for pattern_len in range(1, true_max_L + 1):
        for offset in range(pattern_len):
            end = text_len - offset
            if end < pattern_len:
                break

            pattern = text[end - pattern_len : end]
            idx = end - pattern_len
            repeats = 1

            while idx - pattern_len >= 0:
                if text[idx - pattern_len : idx] == pattern:
                    repeats += 1
                    idx -= pattern_len
                else:
                    break

            total_chars = repeats * pattern_len
            if total_chars >= threshold_chars and repeats >= 2:
                cut_point = idx + pattern_len
                trimmed = text[:cut_point]
                return trimmed, cut_point, pattern_len

    return text, None, None
