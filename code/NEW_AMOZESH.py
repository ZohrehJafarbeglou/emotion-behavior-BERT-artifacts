import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import joblib
from collections import Counter

# تعریف دستگاه (GPU یا CPU) در ابتدای کد
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# بارگذاری داده‌ها از فایل اکسل
df = pd.read_excel('LAST_merged_file_with_BSI.xlsx')

# حذف ردیف‌هایی که ستون 'description' آن‌ها NaN است
df = df.dropna(subset=['description'])

# تبدیل برچسب‌های emotion به اعداد
label_encoder_emotion = LabelEncoder()
df['emotion_label'] = label_encoder_emotion.fit_transform(df['emotion'])

# تعریف رفتارها
behavior_labels = ['ترسیده', 'محتاط', 'منفعل', 'مضطرب', 'اجتنابی', 'متفکر',
                  'خوش‌بین', 'کنجکاو', 'اجتماعی', 'انگیزه‌مند', 'ریسک‌پذیر', 'طنزآمیز']
def encode_behaviors(behaviors):
    return [1 if b in eval(behaviors) else 0 for b in behavior_labels]

df['behavior_labels'] = df['behavior'].apply(encode_behaviors)

# بررسی تعادل داده‌ها
emotion_counts = Counter(df['emotion_label'])
print("Distribution of emotion labels:", {label: count for label, count in zip(label_encoder_emotion.classes_, emotion_counts.values())})

# محاسبه وزن‌ها برای تعادل کلاس‌ها
num_classes = len(label_encoder_emotion.classes_)
class_weights = [max(emotion_counts.values()) / count if count > 0 else 1.0 for count in emotion_counts.values()]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)  # استفاده از device که حالا تعریف شده
print("Class weights:", class_weights)

# جداسازی داده‌ها
X = df['description'].values
y_emotion = df['emotion_label'].values
y_behavior = df['behavior_labels'].values

# جداسازی داده‌ها با stratify
X_train, X_test, y_emotion_train, y_emotion_test, y_behavior_train, y_behavior_test = train_test_split(
    X, y_emotion, y_behavior, test_size=0.2, random_state=42, stratify=y_emotion
)

# بارگذاری توکنایزر و مدل BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=8 + len(behavior_labels)  # اطمینان از 20 کلاس
)

# تنظیمات دستگاه (تکرار برای اطمینان)
model.to(device)

# کلاس دیتاست
class MovieDataset(Dataset):
    def __init__(self, summaries, emotion_labels, behavior_labels, tokenizer, max_len=128):
        self.summaries = summaries
        self.emotion_labels = emotion_labels
        self.behavior_labels = behavior_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        summary = self.summaries[idx]
        emotion_label = self.emotion_labels[idx]
        behavior_label = torch.FloatTensor(self.behavior_labels[idx])

        encoding = self.tokenizer.encode_plus(
            summary,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'emotion_labels': torch.tensor(emotion_label, dtype=torch.long),
            'behavior_labels': behavior_label
        }

# ایجاد دیتالودرها
train_dataset = MovieDataset(X_train, y_emotion_train, y_behavior_train, tokenizer)
test_dataset = MovieDataset(X_test, y_emotion_test, y_behavior_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# تنظیمات آموزشی
epochs = 20
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
loss_fn_emotion = CrossEntropyLoss(weight=class_weights)
loss_fn_behavior = BCEWithLogitsLoss()

best_loss = float('inf')
patience = 3
trigger_times = 0

# فرآیند آموزش
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        emotion_labels = batch['emotion_labels'].to(device)
        behavior_labels = batch['behavior_labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        emotion_logits = logits[:, :8]
        behavior_logits = logits[:, 8:20]  # صریحاً 12 رفتار را جدا کنید
        
        loss_emotion = loss_fn_emotion(emotion_logits, emotion_labels)
        loss_behavior = loss_fn_behavior(behavior_logits, behavior_labels)
        loss = loss_emotion + 0.5 * loss_behavior
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}')
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        trigger_times = 0
        model.save_pretrained('./best_model_emotion_behavior_optimized_1')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered!")
            break

tokenizer.save_pretrained('./best_model_emotion_behavior_optimized_1')
print("Model and tokenizer saved successfully!")

# ارزیابی مدل
model.eval()
emotion_predictions = []
behavior_predictions = []
true_emotion_labels = []
true_behavior_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        emotion_labels = batch['emotion_labels'].to(device)
        behavior_labels = batch['behavior_labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        emotion_logits = logits[:, :8]
        behavior_logits = logits[:, 8:20]
        
        _, emotion_preds = torch.max(emotion_logits, dim=1)
        behavior_preds = torch.sigmoid(behavior_logits) > 0.5
        
        emotion_predictions.extend(emotion_preds.cpu().numpy())
        behavior_predictions.extend(behavior_preds.cpu().numpy())
        true_emotion_labels.extend(emotion_labels.cpu().numpy())
        true_behavior_labels.extend(behavior_labels.cpu().numpy())

emotion_accuracy = accuracy_score(true_emotion_labels, emotion_predictions)
behavior_f1 = f1_score(true_behavior_labels, behavior_predictions, average='micro', zero_division=0)
print(f'Emotion Accuracy: {emotion_accuracy:.2f}')
print(f'Behavior F1 Score: {behavior_f1:.2f}')

# تابع پیش‌بینی
def predict_emotion_and_behavior(summary):
    model.eval()
    inputs = tokenizer(summary, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    emotion_logits = logits[:, :8]
    behavior_logits = logits[:, 8:20]  # صریحاً 12 رفتار
    
    emotion_probs = torch.softmax(emotion_logits, dim=-1).cpu().numpy()[0]
    emotion_classes = label_encoder_emotion.classes_
    emotion_percentages = {emotion: f"{prob * 100:.1f}%" for emotion, prob in zip(emotion_classes, emotion_probs)}
    _, emotion_pred = torch.max(emotion_logits, dim=1)
    predicted_emotion_idx = emotion_pred.cpu().numpy()[0]
    predicted_emotion = emotion_classes[predicted_emotion_idx] if predicted_emotion_idx < len(emotion_classes) else "نامشخص"
    
    behavior_probs = torch.sigmoid(behavior_logits).cpu().numpy()[0][:12]  # فقط 12 رفتار
    behavior_percentages = {behavior_labels[i]: f"{prob * 100:.1f}%" for i, prob in enumerate(behavior_probs)}
    
    if predicted_emotion != "نامشخص":
        if predicted_emotion == "Fear":
            relevant_behaviors = ['ترسیده', 'محتاط', 'منفعل']
        elif predicted_emotion == "Joy":
            relevant_behaviors = ['خوش‌بین', 'اجتماعی', 'انگیزه‌مند']
        elif predicted_emotion == "Sadness":
            relevant_behaviors = ['منفعل', 'متفکر', 'اجتنابی']
        else:
            relevant_behaviors = behavior_labels

        relevant_probs = [behavior_percentages[b] for b in relevant_behaviors if float(behavior_percentages[b][:-1]) > 0]
        total_prob = sum(float(p[:-1]) for p in relevant_probs) if relevant_probs else 1.0
        adjusted_behavior_percentages = {}
        for behavior in behavior_labels:
            if behavior in relevant_behaviors and float(behavior_percentages[behavior][:-1]) > 0:
                adjusted_prob = (float(behavior_percentages[behavior][:-1]) / total_prob) * 100
                adjusted_behavior_percentages[behavior] = f"{min(adjusted_prob, 100):.1f}%"
            else:
                adjusted_behavior_percentages[behavior] = f"{0.0}%"
    else:
        adjusted_behavior_percentages = behavior_percentages

    behavior_preds = [label for label, prob in adjusted_behavior_percentages.items() if float(prob[:-1]) > 10]
    
    return predicted_emotion, emotion_percentages, adjusted_behavior_percentages, behavior_preds

# دریافت ورودی و پیش‌بینی
print("لطفاً متن فیلمنامه یا خلاصه‌ای از آن را وارد کنید (برای پایان، خط خالی وارد کنید):")
lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)
summary = "\n".join(lines)

predicted_emotion, emotion_percentages, behavior_percentages, predicted_behaviors = predict_emotion_and_behavior(summary)

print(f'Predicted Emotion: {predicted_emotion}')
print('Emotion Percentages:')
for emotion, percentage in emotion_percentages.items():
    print(f'{emotion}: {percentage}')
print('Behavior Percentages:')
for behavior, percentage in behavior_percentages.items():
    print(f'{behavior}: {percentage}')
print(f'Predicted Behaviors (above 10%): {predicted_behaviors}')

plt.figure(figsize=(10, 5))
plt.bar(emotion_percentages.keys(), [float(p[:-1]) for p in emotion_percentages.values()])
plt.title('Percentages of Emotions')
plt.xlabel('Emotions')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(behavior_percentages.keys(), [float(p[:-1]) for p in behavior_percentages.values()])
plt.title('Percentages of Behaviors')
plt.xlabel('Behaviors')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

joblib.dump(label_encoder_emotion, './label_encoder_emotion_optimized.pkl')
print("Model, tokenizer, and label_encoder saved successfully!")
