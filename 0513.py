### pipeline

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!(really???)"
    ]
)


## Transformer 모델도 다른 신경망과 마찬가지로 원시 텍스트를 직접 처리할 수 없다.
## 즉, 텍스트 입력을 모델이 이해할 수 있는 숫자로 변환하는 과정이 필요하다.

## tokenizer : Slice, Mapping, Additional info.

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  ## default checkpoint of sentiment-analysis pipeline
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  ## create tokenizer

## 토크나이저가 반환하는 텐서의 유형 지정

raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!(really???)"
]
inputs = tokenizer(raw_inputs, padding = True, truncation = True, return_tensors = "pt")  ## return as torch.tensor
print(inputs)

print(inputs.__class__)
