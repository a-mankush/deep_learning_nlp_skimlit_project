import tensorflow as tf
from spacy.lang.en import English

nlp = English()
sentencizer = nlp.add_pipe("sentencizer")


def list_to_dict(abstract_lines: list):
    pred_list = []
    total_len = len(abstract_lines)
    for i, sent in enumerate(abstract_lines):
        pred_dict = {"text": sent, 'line_number': i, 'total_line': total_len}
        pred_list.append(pred_dict)
    return pred_list


def split_chars(text):
    return " ".join(list(text))


def line_no_ohe(pred_list):
    test_abstract_line_number = [line['line_number'] for line in pred_list]
    return tf.one_hot(test_abstract_line_number, depth=16)


def total_line_ohe(pred_list):
    test_abstract_total_lines = [line["total_line"] for line in pred_list]
    # One-hot encode to same depth as training data, so model accepts right input shape
    return tf.one_hot(test_abstract_total_lines, depth=20)


def abstract_to_sentence(abstract):
    doc = nlp(abstract)
    return [str(sent) for sent in list(doc.sents)]


def preprocess_text(text):
    abstract_lines = abstract_to_sentence(text)
    print(abstract_lines)
    pred_list = list_to_dict(abstract_lines)
    test_abstract_line_numbers_one_hot = line_no_ohe(pred_list)
    test_abstract_total_lines_one_hot = total_line_ohe(pred_list)
    abstract_chars = [split_chars(sent) for sent in abstract_lines]

    return (
        tf.constant(abstract_lines),
        tf.constant(abstract_chars),
        test_abstract_line_numbers_one_hot,
        test_abstract_total_lines_one_hot
    )


def get_predictions_labels(predictions_values):
    class_names = ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']
    return [class_names[i] for i in tf.argmax(predictions_values, axis=1)]
