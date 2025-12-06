Emotion-Behavior BERT Artifacts
This repository contains all artifacts related to the Emotion-Behavior BERT study, including datasets, code, and figures used for emotion and behavior analysis in Persian movie scripts.

Repository Structure
data/
Contains the datasets used for training and testing the model:

LAST_merged_file_with_BSI.xlsx – Main dataset with emotion and behavior labels.

LAST_merged_file_with_main dataset.xlsx – Supplementary dataset with additional data points.

code/
Contains the Python scripts for preprocessing, training, evaluation, and predictions using the BERT-based model:

NEW_AMOZESH.py – Full implementation of the Emotion-Behavior BERT model including training, evaluation, and prediction functions.

figures/
Visualizations generated from the dataset and model predictions:

emotion_distribution.png – Distribution of emotion labels in the dataset.

behavior_distribution.png – Distribution of behavior labels in the dataset.

predicted_emotion_percentages.png – Predicted emotion percentages for a sample summary.

predicted_behavior_percentages.png – Predicted behavior percentages for a sample summary.

Drawing2.pdf, mafhoumi.png, model.png, model 1.png – Additional figures illustrating model structure and results.
Usage
Clone the repository:
git clone https://github.com/ZohrehJafarbeglou/emotion-behavior-BERT-artifacts.git
Install required Python packages (recommend using a virtual environment):
pip install -r requirements.txt
Place your dataset files in the data/ folder and run the Python scripts in the code/ folder to train or evaluate the model.

Citation
If you use this repository for research, please cite our work:
Khadyor, A., Omman, P., & Abbasi, F. (2023). Analysis of Twitter Users' Sentiments on ChatGPT Technology. Information Management, 9(1), 159-182. https://doi.org/10.22034/AIMJ.2024.431651.1576


# شواهد پروژه BERT احساس-رفتار (Emotion-Behavior BERT Artifacts)

این مخزن شامل تمام شواهد (Artifacts) مربوط به مطالعه مدل **BERT احساس-رفتار** است. در این پروژه، داده‌ها و مدل برای تحلیل احساسات و رفتارها در فیلمنامه‌های فارسی استفاده شده است. این مخزن شامل سه بخش اصلی است: **داده‌ها، کدها، و نمودارها**.

## ساختار مخزن

### ۱. پوشه داده‌ها (`data/`)
این پوشه شامل دیتاست‌های مورد استفاده برای آموزش و ارزیابی مدل است:

- `LAST_merged_file_with_BSI.xlsx`  
  دیتاست اصلی که شامل متن‌ها و برچسب‌های احساس و رفتار است.  
- `LAST_merged_file_with_main dataset.xlsx`  
  دیتاست تکمیلی با داده‌های اضافی برای افزایش حجم و تنوع داده‌ها.

> توضیح: داده‌ها به صورت Excel هستند و قبل از آموزش مدل، پردازش اولیه روی آن‌ها انجام شده است.

---

### ۲. پوشه کدها (`code/`)
این پوشه شامل اسکریپت‌های پایتون برای پیش‌پردازش داده‌ها، آموزش مدل، ارزیابی و پیش‌بینی است:

- `NEW_AMOZESH.py`  
  پیاده‌سازی کامل مدل BERT احساس-رفتار شامل:  
  - پردازش داده‌ها و تبدیل برچسب‌ها به فرمت قابل استفاده توسط مدل  
  - آموزش مدل با استفاده از BERT چندزبانه  
  - ارزیابی مدل با محاسبه دقت و F1-score  
  - توابع پیش‌بینی احساس و رفتار برای متون جدید  

> توجه: اجرای کد نیاز به نصب کتابخانه‌های پایتون مانند `transformers`، `torch`، `scikit-learn` و `pandas` دارد.

---

### ۳. پوشه نمودارها (`figures/`)
این پوشه شامل نمودارها و تصاویر تولید شده از داده‌ها و پیش‌بینی‌های مدل است:

- `emotion_distribution.png`  
  نمودار توزیع برچسب‌های احساس در دیتاست.  
- `behavior_distribution.png`  
  نمودار توزیع برچسب‌های رفتار در دیتاست.  
- `predicted_emotion_percentages.png`  
  نمودار درصدهای پیش‌بینی احساس برای یک نمونه متن.  
- `predicted_behavior_percentages.png`  
  نمودار درصدهای پیش‌بینی رفتار برای یک نمونه متن.  
- `Drawing2.pdf`, `mafhoumi.png`, `model.png`, `model 1.png`  
  تصاویر و نمودارهای اضافی مربوط به ساختار مدل و نتایج آن.

> توضیح: این نمودارها برای تحلیل دقیق رفتار مدل و نمایش نتایج آن استفاده شده‌اند.

---

## نحوه استفاده

۱. کلون کردن مخزن:
```bash
git clone https://github.com/ZohrehJafarbeglou/emotion-behavior-BERT-artifacts.git
```

۲. نصب بسته‌های مورد نیاز پایتون:
```bash
pip install -r requirements.txt
```

۳. قرار دادن فایل‌های دیتاست در پوشه `data/` و اجرای اسکریپت‌ها از پوشه `code/` برای آموزش و ارزیابی مدل.

۴. مشاهده نمودارها در پوشه `figures/` برای تحلیل نتایج و مقایسه پیش‌بینی‌ها.

---

## نحوه استناد

در صورت استفاده از این مخزن در تحقیقات خود، لطفاً به مقاله یا پروژه زیر ارجاع دهید:

```
Khadyor, A., Omman, P., & Abbasi, F. (2023). Analysis of Twitter Users' Sentiments on ChatGPT Technology. Information Management, 9(1), 159-182. https://doi.org/10.22034/AIMJ.2024.431651.1576
```

