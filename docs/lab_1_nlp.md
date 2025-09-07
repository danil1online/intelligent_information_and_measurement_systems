# Методические указания к лабораторной работе №1 
**Тема:** Классификация и понимание текста. 2 часа

---

## 🎯 Цель работы  
Освоить ключевые метрики оценки качества текстовой классификации и методы измерения семантического сходства на примерах общедоступных датасетов.

---

## 📌 Задачи  
- Изучить метрики Accuracy, Precision, Recall, F1-Score на основе бинарного датасета 20 Newsgroups.
- Изучить macro-/micro-averaging на многоклассовом датасете 20 Newsgroups.
- Изучить Cosine, Jaccard и Euclidean меры сходства на датасете STS-Benchmark (Semantic Textual Similarity).
- Получить эмбеддинги через BERT/RoBERTa и сравнить качество косинусного сходства с TF-IDF.
- Визуализировать текстовые эмбеддинги методом t-SNE и UMAP.

---

## 📁 Материалы и методы
- Язык программирования – Python 3.10.
- Основные библиотеки:
  - [matplotlib](https://matplotlib.org/),
  - [PyTorch (torch, torchtext)](https://pytorch.org/),
  - [datasets (Hugging Face)](https://huggingface.co/docs/datasets/index),
  - [transformers](https://huggingface.co/docs/transformers/index),
  - [seaborn](https://seaborn.pydata.org/)
  - [umap-learn](https://github.com/lmcinnes/umap).
- Датасеты
  - [20 Newsgroups (sklearn):](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html): бинарная классификация: категории `alt.atheism` vs `soc.religion.christian`; многоклассовая: категории `alt.atheism`, `comp.graphics`, `sci.med`, `soc.religion.christian`
или
  - STS-Benchmark (Hugging Face “glue”, “stsb”): пары предложений с оценками схожести (0–5).

---

## 📚 Краткая теоретическая информация  

### 📚 Классификация текстов и метрики качества

В задаче бинарной или многоклассовой классификации текста используют следующие основные метрики:

1. 📚 Бинарная классификация

1.1. 📚 Accuracy — доля правильно классифицированных образцов:

$$
  Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

1.2. 📚 Precision — точность положительного предсказания:

$$
  Precision = \frac{TP}{TP + FP}
$$

1.3. 📚 Recall — полнота обнаружения позитивных образцов:

$$
  Recall = \frac{TP}{TP + FN}
$$

1.4. 📚 F1-Score — гармоническое среднее Precision и Recall:

$$
  F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

Здесь TP, TN, FP, FN — числа истинно/ложно положительных и отрицательных предсказаний.

2. 📚 Macro- и Micro-Averaging

Используются для оценки качества классификации в многоклассовых задачах.

2.1. 📚 Micro-Averaging

- Суммирует все TP, FP, FN по классам, затем вычисляет метрику.
- Формула для Precision:

$$
  Precision_{micro} = \frac{\sum TP}{\sum TP + \sum FP}
$$

Аналогично для Recall и F1.

2.2. 📚 Macro-Averaging

- Вычисляет метрику **по каждому классу**, затем усредняет.
- Формула для Precision:

$$
  Precision_{macro} = \frac {1}{N} \sum_{i=1}^N Precision_i
$$



### 📚 Cходство

1. 📚 Cosine Similarity измеряет угол между двумя векторными представлениями текстов. Для двух векторов 𝐴 и 𝐵:

$$
  CosSim(A,B) = \frac{A \cdot B}{\Vert A \Vert \cdot \Vert B \Vert}
$$

Принимает значение из диапазона от −1 до +1, где +1 — полное совпадение направлений.

2. 📚 Jaccard Similarity

- Мера схожести между множествами:

$$
  J(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

- Jaccard Distance:

$$
  D_J(A,B) = 1 - J(A,B)
$$

3. 📚 Euclidean Distance

- Классическая метрика расстояния:

$$
  d(p,q) = \sqrt {\sum_{i=1}^n (p_i - q_i)^2}
$$

- Используется в кластеризации, KNN, PCA и др.

### 📚 BERT и RoBERTa

Обе модели — трансформеры для обработки естественного языка.

1. 📚 BERT (Bidirectional Encoder Representations from Transformers)
- Обучается на:
  - [Masked Language Modeling (MLM)](https://huggingface.co/docs/transformers/tasks/masked_language_modeling)
  - [Next Sentence Prediction (NSP)](https://arxiv.org/abs/2109.03564)
- Архитектура: энкодер трансформера с двунаправленным вниманием.

2. 📚 RoBERTa
- Улучшенная версия BERT:
  - Удалён NSP
  - Больше данных и эпох
  - Динамическое маскирование
- Архитектура та же, но обучение более "агрессивное".

### 📚 TF-IDF (Term Frequency–Inverse Document Frequency)

Метод взвешивания слов в тексте. Используется в поиске, классификации, кластеризации текста.

1. 📚 Term Frequency (TF):

$$
  TF(t,d) = \frac {f_{t,d}}{\sum_{t'} f_{t',d}}
$$

2. 📚 Inverse Document Frequency (IDF):

$$
  IDF(t) = \log \frac {N}{1 + df_t}
$$

3. 📚 TF-IDF:

$$
  TFIDF(t,d) = TF(t,d) \times IDF(t)
$$

### 📚 t-SNE и UMAP

Методы снижения размерности, особенно широков распространены для визуализации. UMAP быстрее и лучше масштабируется, чем t-SNE.

1. 📚 t-SNE (t-distributed Stochastic Neighbor Embedding)
- Сохраняет локальные структуры данных.
- Шаги:
  - Вычисление вероятностей сходства в высоком пространстве
  - Инициализация точек в 2D/3D
  - Минимизация KL-дивергенции между распределениями

2. 📚 UMAP (Uniform Manifold Approximation and Projection)
- Основан на теории топологических многообразий.
- Шаги:
  - Построение графа ближайших соседей
  - Формирование "fuzzy" топологии
  - Оптимизация в низком пространстве через SGD


---
 
## ⚙️ Настройка среды

0. Подключитесь к [Jupyter-Hub-ИИСТ-НПИ](http://195.133.13.56:8000/) из [первой работы](docs/lab_1_cv_metrics.md#%EF%B8%8F-настройка-среды)
1. Создайте в корне домашнего каталога каталог проекта и перейдите в него:
```bash

mkdir text_lab && cd text_lab
mkdir results

```
2. Создайте и активируйте виртуальное окружение:
```bash

python3.10 -m venv venv
source venv/bin/activate

```

3. Установите зависимости:
```bash

pip install --upgrade pip setuptools wheel
pip install torch torchtext --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib transformers datasets umap-learn pandas seaborn


```

4. Загрузка датасета

```python

#!/usr/bin/env python3
# file: download_data.py

import os
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from datasets import load_dataset

os.makedirs('data', exist_ok=True)

# 1. 20 Newsgroups: бинарная классификация
cats_bin = ['alt.atheism', 'soc.religion.christian']
train_bin = fetch_20newsgroups(subset='train', categories=cats_bin,
                              remove=('headers','footers','quotes'))
test_bin  = fetch_20newsgroups(subset='test',  categories=cats_bin,
                              remove=('headers','footers','quotes'))

pd.DataFrame({'text': train_bin.data, 'label': train_bin.target}) \
  .to_csv('data/news_train_binary.csv', index=False)
pd.DataFrame({'text': test_bin.data,  'label': test_bin.target}) \
  .to_csv('data/news_test_binary.csv', index=False)

# 2. 20 Newsgroups: многоклассовая классификация
cats_multi = ['alt.atheism','comp.graphics','sci.med','soc.religion.christian']
train_m = fetch_20newsgroups(subset='train', categories=cats_multi,
                            remove=('headers','footers','quotes'))
test_m  = fetch_20newsgroups(subset='test',  categories=cats_multi,
                            remove=('headers','footers','quotes'))

pd.DataFrame({'text': train_m.data, 'label': train_m.target}) \
  .to_csv('data/news_train_multi.csv', index=False)
pd.DataFrame({'text': test_m.data,  'label': test_m.target}) \
  .to_csv('data/news_test_multi.csv', index=False)

# 3. STS-Benchmark: семантическое сходство
sts = load_dataset('glue', 'stsb')
for split in ['train','validation','test']:
    df = pd.DataFrame({
        'sentence1': sts[split]['sentence1'],
        'sentence2': sts[split]['sentence2'],
        'score':     sts[split]['label']
    })
    df.to_csv(f'data/stsb_{split}.csv', index=False)

print("Datasets downloaded to ./data/")


```

---

## 🧪 Примеры

### 🧪 Бинарная классификация (20 Newsgroups)

**Плюсы**: быстрая настройка, понятные метрики. 
**Минусы**: чувствительна к качеству TF-IDF, сложно масштабировать на нерегулярные тексты.

```python
#!/usr/bin/env python3
# file: classify_binary.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Загрузка
train = pd.read_csv('data/news_train_binary.csv')
test  = pd.read_csv('data/news_test_binary.csv')

# Удаление NaN и пустых строк
train = train.dropna(subset=['text'])
test = test.dropna(subset=['text'])
train = train[train.text.str.strip().astype(bool)]
test = test[test.text.str.strip().astype(bool)]

vect = TfidfVectorizer(max_features=2000)
X_train = vect.fit_transform(train.text)
X_test  = vect.transform(test.text)
y_train = train.label
y_test  = test.label

# Обучение
clf = LogisticRegression(max_iter=300)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Метрики
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
print(f"Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

# Сохранение графика
plt.bar(['Acc','Prec','Rec','F1'], [acc,prec,rec,f1], color=['#4C72B0','#55A868','#C44E52','#8172B3'])
plt.ylim(0,1); plt.title('Binary Classification Metrics')
plt.savefig('results/binary_metrics.png')

```


Запуск из командной строки:

```bash
python classify_binary.py
```

### 🧪 Пример 2. Многоклассовая классификация с macro/micro

**Плюсы**: микро- и макро-среднее дают разные перспективы при дисбалансе классов. 
**Минусы**: macro не учитывает частоту классов, micro игнорирует качество на редких классах.

```python

#!/usr/bin/env python3
# file: classify_multi.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Загрузка
train = pd.read_csv('data/news_train_multi.csv')
test  = pd.read_csv('data/news_test_multi.csv')

# Удаление NaN и пустых строк
train = train.dropna(subset=['text'])
test = test.dropna(subset=['text'])
train = train[train.text.str.strip().astype(bool)]
test = test[test.text.str.strip().astype(bool)]

vect = TfidfVectorizer(max_features=3000)
X_train = vect.fit_transform(train.text)
X_test  = vect.transform(test.text)
y_train = train.label
y_test  = test.label

# Модель
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Метрики
metrics = {
    'Precision (micro)': precision_score(y_test, y_pred, average='micro'),
    'Recall (micro)':    recall_score(y_test, y_pred, average='micro'),
    'F1 (micro)':        f1_score(y_test, y_pred, average='micro'),
    'Precision (macro)': precision_score(y_test, y_pred, average='macro'),
    'Recall (macro)':    recall_score(y_test, y_pred, average='macro'),
    'F1 (macro)':        f1_score(y_test, y_pred, average='macro'),
}

for name, val in metrics.items():
    print(f"{name}: {val:.3f}")

# Визуализация
plt.bar(metrics.keys(), metrics.values(), color=plt.cm.tab20.colors)
plt.xticks(rotation=45, ha='right'); plt.ylim(0,1)
plt.title('Multi-Class micro/macro Metrics')
plt.tight_layout()
plt.savefig('results/multi_metrics.png')

```

Запуск:

```bash

python classify_multi.py

```

### 🧪 Пример 3. Меры семантического сходства (Jaccard, Euclidean, Cosine)

Плюсы:

  - Jaccard прост, но дискретен.
  - Euclidean чувствителен к длине.
  - Cosine устойчива к масштабированию.

```python

#!/usr/bin/env python3
# file: semantic_measures.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# Берём небольшую выборку из валидации STS-B
df = pd.read_csv('data/stsb_validation.csv').sample(200, random_state=0)
s1 = df.sentence1.tolist()
s2 = df.sentence2.tolist()
gold = df.score.values

# TF-IDF векторизация
vect = TfidfVectorizer()
X1 = vect.fit_transform(s1).toarray()
X2 = vect.transform(s2).toarray()

# Вычисления
jacc = [jaccard_score((X1[i]>0).astype(int), (X2[i]>0).astype(int)) for i in range(len(df))]
euc   = [euclidean(X1[i], X2[i]) for i in range(len(df))]
cos   = [cosine_similarity(X1[i:i+1], X2[i:i+1])[0,0] for i in range(len(df))]

# Корреляция с оценкой человека
print("Corr with gold:")
print(f" Jaccard vs score: {np.corrcoef(jacc, gold)[0,1]:.3f}")
print(f" Euclid vs score : {np.corrcoef(euc, gold)[0,1]:.3f}")
print(f" Cosine vs score : {np.corrcoef(cos, gold)[0,1]:.3f}")

# Сохранение графика: Cosine vs Gold
plt.scatter(gold, cos, alpha=0.5)
plt.xlabel('Human score'); plt.ylabel('Cosine sim'); plt.title('Cosine vs STS-B')
plt.savefig('results/semantic_cosine.png')
```

Запуск:

```bash

python semantic_measures.py

```

### 🧪 Пример 4. Трансформеры для эмбеддингов и Cosine Similarity

Сохраните в semantic_transformers.py.

```python

#!/usr/bin/env python3
# file: semantic_transformers.py

import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# Настройка
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'bert-base-uncased'
tok   = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Загрузка пары
df = pd.read_csv('data/stsb_validation.csv').sample(200, random_state=1)
sent1 = df.sentence1.tolist()
sent2 = df.sentence2.tolist()
gold  = df.score.values

# Функция получения CLS-эмбеддинга
def embed(texts):
    enc = tok(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    out = model(**enc, return_dict=True).last_hidden_state[:,0,:]
    return out.detach().cpu().numpy()

E1 = embed(sent1)
E2 = embed(sent2)
cos_t = [cosine_similarity(E1[i:i+1], E2[i:i+1])[0,0] for i in range(len(E1))]

# Сравнение с TF-IDF Cosine
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer().fit(sent1 + sent2)
T1 = vec.transform(sent1).toarray(); T2 = vec.transform(sent2).toarray()
cos_tf = [cosine_similarity(T1[i:i+1], T2[i:i+1])[0,0] for i in range(len(T1))]

print("MSE vs human:")
print(" BERT Cosine:", mean_squared_error(gold, cos_t))
print(" TF-IDF Cosine:", mean_squared_error(gold, cos_tf))

```

Запуск:

```bash

python semantic_transformers.py

```

### 🧪 Пример 5. Визуализация текстовых эмбеддингов (t-SNE и UMAP)

**Плюсы**: наглядно выявляются кластеры по темам. 
**Минусы**: t-SNE медленный, UMAP требует подбора параметров.

```python


#!/usr/bin/env python3
# file: visualize_embeddings.py

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Данные: небольшой поднабор 20 Newsgroups (многоклассовый)
df = pd.read_csv('data/news_test_multi.csv').sample(400, random_state=2)
texts = df.text.tolist()
labels = df.label.tolist()

# Эмбеддинги через BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model     = AutoModel.from_pretrained('bert-base-uncased').eval()
def get_embs(texts, labels):
    # Фильтрация пар (text, label)
    filtered = [(str(t), l) for t, l in zip(texts, labels) if isinstance(t, str) and t.strip()]
    texts_clean, labels_clean = zip(*filtered)

    enc = tokenizer(list(texts_clean), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        out = model(**enc).last_hidden_state[:, 0, :]
    return out.numpy(), list(labels_clean)

E, labels = get_embs(texts, labels)

# t-SNE
tsne = TSNE(n_components=2, random_state=0)
E_tsne = tsne.fit_transform(E)

# UMAP
um = umap.UMAP(n_components=2, random_state=0)
E_umap = um.fit_transform(E)

assert E_tsne.shape[0] == len(labels), "Размеры эмбеддингов и меток не совпадают!"

# Рисуем
# Проверка размеров
print(f"E_tsne shape: {E_tsne.shape}")
print(f"labels length: {len(labels)}")

# Приведение labels к нужной длине
if len(labels) != E_tsne.shape[0]:
    print("⚠️ Несоответствие размеров: обрезаем labels")
    labels = labels[:E_tsne.shape[0]]

# Создание DataFrame
df_plot = pd.DataFrame({
    'x': E_tsne[:, 0],
    'y': E_tsne[:, 1],
    'label': labels
})

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# t-SNE scatterplot
sns.scatterplot(data=df_plot, x='x', y='y', hue='label', palette='tab10', s=20, ax=axes[0])
axes[0].set_title('t-SNE Embedding Visualization')
axes[0].legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

# Гистограмма распределения классов
sns.countplot(data=df_plot, x='label', hue='label', palette='tab10', ax=axes[1], legend=False)
axes[1].set_title('Label Distribution')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('results/embeddings_vis.png')
print("Result saved in results/embeddings_vis.png")

```

Запуск:

```bash

python visualize_embeddings.py

```

---
### 📌 Задание для самостоятельной работы

1. Построить ROC-кривую и вычислить AUC для бинарной классификации.
2. Сравнить SVM и LogisticRegression на бинарном и многоклассовом датасетах.
3. Провести корреляционный анализ всех трёх мер сходства (Jaccard, Euclidean, Cosine) с оценками STS-B.
4. Попробовать другие модели из transformers (RoBERTa, DistilBERT) и сравнить MSE.
5. Исследовать влияние параметров UMAP (n_neighbors, min_dist) на кластеризацию.

---

## 💡 Не забудьте выключить текущую среду выполнения программы python (должна пропасть надпись (venv) в начале командной строки):

```bash

deactivate

```


## Вопросы
1. Как macro- и micro-averaging отражают баланс между общим качеством и качеством на редких классах?
2. В каких случаях простые меры (Jaccard, Euclidean) уступают Cosine и трансформерам?
3. Насколько сложнее и ресурсоёмко интегрировать трансформеры по сравнению с TF-IDF?
4. Какие визуализационные приёмы позволяют лучше понять структуру текстовых данных?
5. Как выбор порога влияет на Precision и Recall в задачах с несбалансированными классами?
6. Какие преимущества и недостатки у каждой из рассмотренных моделей классификации?
7. Как величина косинусного сходства соотносится с субъективной близостью двух текстов?
8. Какие методы векторизации текста оказались наиболее информативными для ваших данных?
