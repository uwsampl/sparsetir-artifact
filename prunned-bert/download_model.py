from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import BertModel


def download_model():
    structured_tokenizer = AutoTokenizer.from_pretrained(
        "madlag/bert-base-uncased-squad1.1-block-sparse-0.07-v1"
    )
    structured_model = AutoModelForQuestionAnswering.from_pretrained(
        "madlag/bert-base-uncased-squad1.1-block-sparse-0.07-v1"
    )
    unstructured_tokenizer = AutoTokenizer.from_pretrained(
        "madlag/bert-base-uncased-squad1.1-block-sparse-0.07-v1"
    )
    unstructured_model = AutoModelForQuestionAnswering.from_pretrained(
        "madlag/bert-base-uncased-squad1.1-block-sparse-0.07-v1"
    )


if __name__ == "__main__":
    download_model()
