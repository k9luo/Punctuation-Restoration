import codecs
import numpy as np

from utils.data_utils import SPACE, PUNCTUATION_VOCABULARY, PUNCTUATION_MAPPING


def compute_score(target_path, predicted_path):
    """Computes and prints the overall classification error and precision, recall, F-score over punctuations."""
    mappings, counter, t_i, p_i = {}, 0, 0, 0
    total_correct, correct, substitutions, deletions, insertions = 0, 0.0, 0.0, 0.0, 0.0
    true_pos, false_pos, false_neg = {}, {}, {}

    with codecs.open(target_path, "r", "utf-8") as f_target, codecs.open(predicted_path, "r", "utf-8") as f_predict:
        target_stream = f_target.read().split()
        predict_stream = f_predict.read().split()

        while True:
            if PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in PUNCTUATION_VOCABULARY:
                # skip multiple consecutive punctuations
                target_punct = " "
                while PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i]) in PUNCTUATION_VOCABULARY:
                    target_punct = PUNCTUATION_MAPPING.get(target_stream[t_i], target_stream[t_i])
                    target_punct = mappings.get(target_punct, target_punct)
                    t_i += 1
            else:
                target_punct = " "

            if predict_stream[p_i] in PUNCTUATION_VOCABULARY:
                predicted_punct = mappings.get(predict_stream[p_i], predict_stream[p_i])
                p_i += 1
            else:
                predicted_punct = " "

            is_correct = target_punct == predicted_punct
            counter += 1
            total_correct += is_correct

            if predicted_punct == " " and target_punct != " ":
                deletions += 1
            elif predicted_punct != " " and target_punct == " ":
                insertions += 1
            elif predicted_punct != " " and target_punct != " " and predicted_punct == target_punct:
                correct += 1
            elif predicted_punct != " " and target_punct != " " and predicted_punct != target_punct:
                substitutions += 1

            true_pos[target_punct] = true_pos.get(target_punct, 0.0) + float(is_correct)
            false_pos[predicted_punct] = false_pos.get(predicted_punct, 0.0) + float(not is_correct)
            false_neg[target_punct] = false_neg.get(target_punct, 0.0) + float(not is_correct)

            assert target_stream[t_i] == predict_stream[p_i] or predict_stream[p_i] == "<unk>", \
                "File: %s \nError: %s (%s) != %s (%s) \nTarget context: %s \nPredicted context: %s" % \
                (target_path, target_stream[t_i], t_i, predict_stream[p_i], p_i,
                 " ".join(target_stream[t_i - 2:t_i + 2]), " ".join(predict_stream[p_i - 2:p_i + 2]))

            t_i += 1
            p_i += 1

            if t_i >= len(target_stream) - 1 and p_i >= len(predict_stream) - 1:
                break

    overall_tp, overall_fp, overall_fn = 0.0, 0.0, 0.0
    out_str = "-" * 46 + "\n"
    out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("PUNCTUATION", "PRECISION", "RECALL", "F-SCORE")

    for p in PUNCTUATION_VOCABULARY:
        if p == SPACE:
            continue

        overall_tp += true_pos.get(p, 0.0)
        overall_fp += false_pos.get(p, 0.0)
        overall_fn += false_neg.get(p, 0.0)
        punctuation = p
        precision = (true_pos.get(p, 0.0) / (true_pos.get(p, 0.0) + false_pos[p])) if p in false_pos else np.nan
        recall = (true_pos.get(p, 0.0) / (true_pos.get(p, 0.0) + false_neg[p])) if p in false_neg else np.nan
        f_score = (2. * precision * recall / (precision + recall)) if (precision + recall) > 0 else np.nan
        out_str += u"{:<16} {:<9} {:<9} {:<9}\n".format(punctuation, "{:.2f}".format(precision * 100),
                                                        "{:.2f}".format(recall * 100),
                                                        "{:.2f}".format(f_score * 100))
    out_str += "-" * 46 + "\n"
    pre = overall_tp / (overall_tp + overall_fp) if overall_fp else np.nan
    rec = overall_tp / (overall_tp + overall_fn) if overall_fn else np.nan
    f1 = (2.0 * pre * rec) / (pre + rec) if (pre + rec) else np.nan
    out_str += "{:<16} {:<9} {:<9} {:<9}\n".format("Overall", "{:.2f}".format(pre * 100),
                                                   "{:.2f}".format(rec * 100), "{:.2f}".format(f1 * 100))
    err = round((100.0 - float(total_correct) / float(counter - 1) * 100.0), 2)
    ser = round((substitutions + deletions + insertions) / (correct + substitutions + deletions) * 100, 1)
    out_str += "ERR: %s%%\n" % err
    out_str += "SER: %s%%" % ser
    return out_str, f1, err, ser
