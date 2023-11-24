from transformers import BertForSequenceClassification
import torch
model = BertForSequenceClassification.from_pretrained("./results/checkpoint-9375")
model.eval()  # 将模型设置为评估模式
if torch.cuda.is_available():
    model.cuda()  # 如果可用，将模型移至GPU
else:
    model.cpu()  # 否则使用CPU
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 确保输入数据在正确的设备上
    with torch.no_grad():  # 在评估模式下，不追踪梯度
        outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return "Positive" if prediction == 1 else "Negative"

print(predict_sentiment("Star Trek is one of the best works of director Christopher Nolan. It is more important than the chronicle he created in memory fragments, more impressive than the distorted plot in inception, and more daunting than re imagining Batman in Batman: the mystery of the Xia shadow as the most unique superhero Series in the 21st century. The film is not only a high-cost science fiction, but also a simple story about love and sacrifice. It includes impatience, adventure, hope and heartbreak. The creation of the film is an amazing achievement, which is worth watching on the largest screen and using the best sound system."))


