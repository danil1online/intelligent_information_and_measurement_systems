# Методические указания к практической работе №4 
**Тема:** Метрики оценки качества распознавания речи. 4 часа  

---

## 🎯 Цель работы  
Изучить метрики оценки качества решения задачи автоматического распознавания речи (ASR) – WER (Word Error Rate) и CER (Character Error Rate).

---

## 📌 Задачи  
- Ознакомиться с определением и формулами WER и CER.
- Реализовать простой ASR-конвейер на Python с подсчётом метрик.
- На практике рассчитать WER и CER для готовых распознанных текстов.  
- Проанализировать результаты и сформулировать выводы.

---

## 📁 Материалы и методы
- Язык программирования – Python 3.10.
- Основные библиотеки:
  - [matplotlib](https://matplotlib.org/),
  - [PyTorch (torch, torchvision)](https://pytorch.org/),
  - [jiwer](https://github.com/jitsi/jiwer),
  - [SpeechRecognition](https://github.com/Uberi/speech_recognition#readme),
  - [Tensorflow](tensorflow.org),
  - [soundfile](https://github.com/bastibe/python-soundfile),
  - [librosa](https://librosa.org/doc/latest/index.html),
  - [transformers](https://huggingface.co/docs/transformers/index),
  - [openai-whisper](https://github.com/openai/whisper).
- Датасеты
  - [Mozilla Common Voice (русский), выбрать ~100 коротких записей](https://commonvoice.mozilla.org/ru)
или
  - [LibriSpeech “dev-clean” (15–20 предложение)](https://www.openslr.org/12).

Для примера рассмотрим LibriSpeech

---

## 📚 Краткая теоретическая информация  

Автоматическое распознавание речи (ASR) превращает аудиосигнал в текст. Ключевая сложность задачи – шумы, акценты, вариативность произношения.

Метрика WER показывает долю ошибок на уровне слов. Её формула:

$$
  WER = \frac{S + D + I}{N},
$$

где 𝑆 – количество замен, 𝐷 – удалений, 𝐼 – вставок, 𝑁 – число слов в эталонном тексте.

Метрика CER отражает ошибки на символах:

$$
  CER = \frac{S_c + D_c + I_c}{N_c},
$$

где индексы 𝑐 обозначают подсчёт на уровне символов.

## ⚙️ Настройка среды

0. Подключитесь к [Jupyter-Hub-ИИСТ-НПИ](http://195.133.13.56:8000/) из [первой работы](docs/lab_1_cv_metrics.md#%EF%B8%8F-настройка-среды)
1. Создайте в корне домашнего каталога каталог проекта и перейдите в него:
```bash

mkdir lab_asr && cd lab_asr

```
2. Создайте и активируйте виртуальное окружение:
```bash

python3.10 -m venv venv
source venv/bin/activate

```

3. Установите зависимости:
```bash

pip install --upgrade pip setuptools wheel cython
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib tensorflow-cpu jiwer SpeechRecognition soundfile librosa transformers openai-whisper


```

4. (Выполнено) Загрузка датасета

```python

#!/usr/bin/env python3
# File: fetch_librispeech_subset.py

import os
import sys
import urllib.request
import tarfile
import subprocess

# 1) Настройки: URL набора и размер выборки
LIBRISPEECH_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
ARCHIVE_NAME   = "dev-clean.tar.gz"
MAX_SAMPLES    = 10    # сколько фрагментов взять

# 2) Директории для хранения
BASE_DIR       = os.path.abspath(os.path.dirname(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
AUDIO_DIR      = os.path.join(DATA_DIR, "audio")
TRANS_DIR      = os.path.join(DATA_DIR, "transcripts")

for d in (DATA_DIR, AUDIO_DIR, TRANS_DIR):
    os.makedirs(d, exist_ok=True)

# 3) Скачиваем архив, если он ещё не загружен
archive_path = os.path.join(BASE_DIR, ARCHIVE_NAME)
if not os.path.isfile(archive_path):
    print(f"Downloading {LIBRISPEECH_URL} …")
    urllib.request.urlretrieve(LIBRISPEECH_URL, archive_path)
    print("Download complete.")

# 4) Открываем tar.gz и собираем нужные .flac
print(f"Extracting up to {MAX_SAMPLES} audio files …")
ids = []
with tarfile.open(archive_path, "r:gz") as tar:
    for member in tar.getmembers():
        if member.name.endswith(".flac") and len(ids) < MAX_SAMPLES:
            fname = os.path.basename(member.name)
            utt_id = os.path.splitext(fname)[0]
            ids.append(utt_id)

            # извлечение FLAC в файл
            flac_bytes = tar.extractfile(member).read()
            flac_path  = os.path.join(AUDIO_DIR, f"{utt_id}.flac")
            with open(flac_path, "wb") as out_f:
                out_f.write(flac_bytes)
            print(f"  ✅ {utt_id}.flac")

            # конвертация в WAV через ffmpeg
            wav_path = os.path.join(AUDIO_DIR, f"{utt_id}.wav")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", flac_path, wav_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"  🔄 {utt_id}.wav")
            except FileNotFoundError:
                sys.exit("ffmpeg not found. Please install ffmpeg and rerun.")
            except subprocess.CalledProcessError:
                print(f"  ⚠️ Ошибка при конвертации {utt_id}.flac → .wav")

# 5) Извлекаем и сохраняем транскрипции
print("Extracting transcripts …")
with tarfile.open(archive_path, "r:gz") as tar:
    for member in tar.getmembers():
        if member.name.endswith(".trans.txt"):
            for line in tar.extractfile(member).read().decode("utf-8").splitlines():
                utt_id, text = line.split(" ", 1)
                if utt_id in ids:
                    txt_path = os.path.join(TRANS_DIR, f"{utt_id}.txt")
                    with open(txt_path, "w", encoding="utf-8") as out_t:
                        out_t.write(text)
                    print(f"  📝 {utt_id}.txt")

print("\nГотово! Структура папок:")
print(f"  {AUDIO_DIR}/  — audio FLAC + WAV")
print(f"  {TRANS_DIR}/ — transcripts TXT")


```

**Вышеприведенный скрипт уже выполнен пользователем `jupyter`, поэтому для доступа к датасету достаточно указывать путь к датасету в его домашнем каталоге.**

---

## 🧪 Примеры и задания 

### 🧪 Простейшее распознавание

1. В первой простейшей демонстрационной программе реализован подсчёт WER и CER для заданных гипотез и эталонов. 

```python
#!/usr/bin/env python3
# file: compute_errors.py

import os
import sys
import time
import speech_recognition as sr
from jiwer import wer, cer
import matplotlib.pyplot as plt

# Пути к данным
BASE_DIR    = os.path.abspath("/home/jupyter/lab_asr/")
AUDIO_DIR   = os.path.join(BASE_DIR, "data", "audio")
TRANS_DIR   = os.path.join(BASE_DIR, "data", "transcripts")

# Инициализация распознавателя
recognizer = sr.Recognizer()

# Списки для эталонов и гипотез
refs, hyps, utt_ids = [], [], []

# Проходим по всем WAV-файлам
for fname in sorted(os.listdir(AUDIO_DIR)):
    if not fname.lower().endswith(".wav"):
        continue

    utt_id, _ = os.path.splitext(fname)
    wav_path  = os.path.join(AUDIO_DIR, fname)
    txt_path  = os.path.join(TRANS_DIR, f"{utt_id}.txt")

    if not os.path.exists(txt_path):
        print(f"⚠️  Пропускаем {utt_id}: нет транскрипции", file=sys.stderr)
        continue

    # Читаем эталон
    with open(txt_path, encoding="utf-8") as f:
        ref = f.read().strip()
    refs.append(ref)
    utt_ids.append(utt_id)

    # Распознаём через Google Web API
    with sr.AudioFile(wav_path) as src:
        audio = recognizer.record(src)
    try:
        hyp = recognizer.recognize_google(audio, language="ru-RU")
    except sr.RequestError as e:
        print(f"❌ Ошибка API при распознавании {utt_id}: {e}", file=sys.stderr)
        hyp = ""
    except sr.UnknownValueError:
        hyp = ""
    hyps.append(hyp)

    print(f"{utt_id}:")
    print(f"  REF: {ref}")
    print(f"  HYP: {hyp}\n")
    # Небольшая пауза, чтобы не превысить лимит запросов
    time.sleep(0.5)

if not refs:
    sys.exit("Нет распознанных фрагментов. Проверьте WAV-файлы и транскрипции.")

# Вычисляем метрики по каждому utterance
wers = [wer(r, h) for r, h in zip(refs, hyps)]
cers = [cer(r, h) for r, h in zip(refs, hyps)]

# Средние метрики
avg_wer = sum(wers) / len(wers)
avg_cer = sum(cers) / len(cers)

print("=== Результаты ===")
print(f"Средний WER: {avg_wer:.3f}")
print(f"Средний CER: {avg_cer:.3f}")

# Визуализация
plt.figure(figsize=(5, 4))
plt.bar(["WER", "CER"], [avg_wer, avg_cer], color=["skyblue", "salmon"])
plt.ylim(0, max(avg_wer, avg_cer) * 1.2)
plt.ylabel("Error rate")
plt.title("Средние WER и CER")
plt.tight_layout()
plt.savefig("error_rates.png", dpi=150)
print("\nГрафик сохранён в error_rates.png")

```


Запуск из командной строки:

```bash
python compute_errors.py
```

Результат достаточно "плачевный" - ошибки крайне высоки:

```bash
python compute_errors.py
2277-149896-0005:
  REF: MANY LITTLE WRINKLES GATHERED BETWEEN HIS EYES AS HE CONTEMPLATED THIS AND HIS BROW MOISTENED
  HYP: 

2277-149896-0006:
  REF: HE COULD ARRANGE THAT SATISFACTORILY FOR CARRIE WOULD BE GLAD TO WAIT IF NECESSARY
  HYP: 

2277-149896-0012:
  REF: SHE HAD NOT BEEN ABLE TO GET AWAY THIS MORNING
  HYP: 

2277-149896-0015:
  REF: HE WENT IN AND EXAMINED HIS LETTERS BUT THERE WAS NOTHING FROM CARRIE
  HYP: 

2277-149896-0018:
  REF: HIS FIRST IMPULSE WAS TO WRITE BUT FOUR WORDS IN REPLY GO TO THE DEVIL
  HYP: is First impose ribe Forward in Reply go to the Java

2277-149896-0021:
  REF: WHAT WOULD SHE DO ABOUT THAT THE CONFOUNDED WRETCH
  HYP: 

2277-149896-0026:
  REF: THE LONG DRIZZLE HAD BEGUN PEDESTRIANS HAD TURNED UP COLLARS AND TROUSERS AT THE BOTTOM
  HYP: The Long restl Happy Gun 15

2277-149896-0027:
  REF: HURSTWOOD ALMOST EXCLAIMED OUT LOUD AT THE INSISTENCY OF THIS THING
  HYP: 

2277-149896-0033:
  REF: THEN HE RANG THE BELL NO ANSWER
  HYP: 

2277-149896-0034:
  REF: HE RANG AGAIN THIS TIME HARDER STILL NO ANSWER
  HYP: Heaven again the time Hunter Still no answer

=== Результаты ===
Средний WER: 1.000
Средний CER: 0.950

График сохранён в error_rates.png
```


2. Во второй простейшей демонстрационной программе распознавание реализовано с использованием также не подходящей модели - через SpeechRecognition + метрики

```python

#!/usr/bin/env python3
# file: asr_pipeline.py

import os
import sys
import time
import speech_recognition as sr
from jiwer import wer, cer
import matplotlib.pyplot as plt

# Пути к данным
BASE_DIR   = os.path.abspath("/home/jupyter/lab_asr/")
AUDIO_DIR  = os.path.join(BASE_DIR, "data", "audio")
TRANS_DIR  = os.path.join(BASE_DIR, "data", "transcripts")

# Инициализация распознавателя
recognizer = sr.Recognizer()

def transcribe(wav_path: str) -> str:
    """
    Считывает аудио из файла и возвращает распознанную строку.
    В случае ошибки возвращает пустую строку и печатает предупреждение.
    """
    with sr.AudioFile(wav_path) as src:
        audio = recognizer.record(src)
    try:
        return recognizer.recognize_google(audio, language="ru-RU")
    except sr.UnknownValueError:
        print(f"⚠️ Не удалось распознать: {os.path.basename(wav_path)}", file=sys.stderr)
        return ""
    except sr.RequestError as e:
        print(f"❌ Ошибка API при распознавании {os.path.basename(wav_path)}: {e}", file=sys.stderr)
        return ""

# Собираем эталоны и гипотезы
refs, hyps, utts = [], [], []
for fname in sorted(os.listdir(AUDIO_DIR)):
    if not fname.lower().endswith(".wav"):
        continue

    utt_id   = os.path.splitext(fname)[0]
    wav_path = os.path.join(AUDIO_DIR, fname)
    txt_path = os.path.join(TRANS_DIR, f"{utt_id}.txt")

    if not os.path.exists(txt_path):
        print(f"⚠️ Пропущено {utt_id}: нет {utt_id}.txt", file=sys.stderr)
        continue

    # Эталон
    with open(txt_path, encoding="utf-8") as f:
        ref = f.read().strip()
    refs.append(ref)
    utts.append(utt_id)

    # Гипотеза
    hyp = transcribe(wav_path)
    hyps.append(hyp)

    print(f"{utt_id}  |  REF: {ref}")
    print(f"         HYP: {hyp}\n")
    time.sleep(0.5)   # чтобы не бить по API слишком быстро

if not refs:
    sys.exit("Ошибка: не найдено ни одного корректного .wav + .txt.")

# Подсчёт WER и CER
wers = [wer(r, h) for r, h in zip(refs, hyps)]
cers = [cer(r, h) for r, h in zip(refs, hyps)]
avg_wer = sum(wers) / len(wers)
avg_cer = sum(cers) / len(cers)

print("=== Итоги распознавания ===")
print(f"Средний WER: {avg_wer:.3f}")
print(f"Средний CER: {avg_cer:.3f}")

# Визуализация результатов
plt.figure(figsize=(5,4))
plt.bar(["WER","CER"], [avg_wer, avg_cer], color=["skyblue","salmon"])
plt.ylim(0, max(avg_wer, avg_cer)*1.2)
plt.ylabel("Error rate")
plt.title("Средние WER и CER")
plt.tight_layout()
plt.savefig("error_rates.png")
print("\nГрафик сохранён: error_rates.png")

```

Полученные результаты также крайне неудовлетворительны:

```bash

python asr_pipeline.py
⚠️ Не удалось распознать: 2277-149896-0005.wav
2277-149896-0005  |  REF: MANY LITTLE WRINKLES GATHERED BETWEEN HIS EYES AS HE CONTEMPLATED THIS AND HIS BROW MOISTENED
         HYP: 

⚠️ Не удалось распознать: 2277-149896-0006.wav
2277-149896-0006  |  REF: HE COULD ARRANGE THAT SATISFACTORILY FOR CARRIE WOULD BE GLAD TO WAIT IF NECESSARY
         HYP: 

⚠️ Не удалось распознать: 2277-149896-0012.wav
2277-149896-0012  |  REF: SHE HAD NOT BEEN ABLE TO GET AWAY THIS MORNING
         HYP: 

⚠️ Не удалось распознать: 2277-149896-0015.wav
2277-149896-0015  |  REF: HE WENT IN AND EXAMINED HIS LETTERS BUT THERE WAS NOTHING FROM CARRIE
         HYP: 

2277-149896-0018  |  REF: HIS FIRST IMPULSE WAS TO WRITE BUT FOUR WORDS IN REPLY GO TO THE DEVIL
         HYP: is First impose ribe Forward in Reply go to the Java

⚠️ Не удалось распознать: 2277-149896-0021.wav
2277-149896-0021  |  REF: WHAT WOULD SHE DO ABOUT THAT THE CONFOUNDED WRETCH
         HYP: 

2277-149896-0026  |  REF: THE LONG DRIZZLE HAD BEGUN PEDESTRIANS HAD TURNED UP COLLARS AND TROUSERS AT THE BOTTOM
         HYP: The Long restl Happy Gun 15

⚠️ Не удалось распознать: 2277-149896-0027.wav
2277-149896-0027  |  REF: HURSTWOOD ALMOST EXCLAIMED OUT LOUD AT THE INSISTENCY OF THIS THING
         HYP: 

⚠️ Не удалось распознать: 2277-149896-0033.wav
2277-149896-0033  |  REF: THEN HE RANG THE BELL NO ANSWER
         HYP: 

2277-149896-0034  |  REF: HE RANG AGAIN THIS TIME HARDER STILL NO ANSWER
         HYP: Heaven again the time Hunter Still no answer

=== Итоги распознавания ===
Средний WER: 1.000
Средний CER: 0.950

График сохранён: error_rates.png

```
WER=1.0 и CER≈0.95 на Google-API при работе с LibriSpeech dev-clean — это ожидаемо: используется телефонная модель Google, рассчитанная на бытовую речь, а LibriSpeech читается в лабораторных условиях с четкой дикцией и англоязычными фрагментами. 

Чтобы получить более адекватные результаты, нужно реализовать один из вариантов (или все):
- Перейти на локальную модель, специально обученную на англоязычной речи «читка»
- Подготовить аудио под требования модели: 16 kHz, моно, VAD-тримминг, нормализация
- Использовать более мощные ASR-архитектуры: Wav2Vec2 или Whisper

---

### 🧪 Пример 2. Улучшение результатов распознавания. Wav2Vec2 (fairseq / HuggingFace)

Ниже пример скрипта `hf_wav2vec2_asr.py`, который использует модель Wav2Vec2

```python

#!/usr/bin/env python3
# file: hf_wav2vec2_asr.py

import os
import torch
import soundfile as sf
from jiwer import wer, cer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self"
AUDIO_DIR = "/home/jupyter/lab_asr/data/audio"
TRANS_DIR = "/home/jupyter/lab_asr/data/transcripts"

# 1) Загрузка модели и процессора
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")

refs, hyps = [], []

for fn in sorted(os.listdir(AUDIO_DIR)):
    if not fn.endswith(".wav"): continue
    utt = os.path.splitext(fn)[0]
    wav, sr = sf.read(os.path.join(AUDIO_DIR, fn))
    # 2) Ресемплинг (если нужно)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    # 3) Препроцессинг: VAD-trim (опционально)
    # wav, _ = librosa.effects.trim(wav, top_db=20)

    # 4) Токенизация и вывод логитов
    input_values = processor(wav, sampling_rate=16000, return_tensors="pt").input_values
    logits       = model(input_values.to(model.device)).logits
    pred_ids     = torch.argmax(logits, dim=-1)
    transcription= processor.batch_decode(pred_ids)[0].lower().strip()

    refs.append(open(os.path.join(TRANS_DIR, f"{utt}.txt"), encoding="utf-8").read().lower().strip())
    hyps.append(transcription)

    print(f"{utt}  |  HYP: {transcription}")

# 5) Подсчёт метрик
print("WER:", wer(refs, hyps))
print("CER:", cer(refs, hyps))

```

Запуск:

```bash

python hf_wav2vec2_asr.py

```

Результат - удовлетворительный. Модель Wav2Vec2ForCTC (имеет значительный объем 1.26G) предупреждает:

```bash
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```
но при этом полученные результаты намного лучше ранее полученных:
```bash
WER: 0.01694915254237288
CER: 0.0062402496099844
```

---

### 🧪 Пример 3. Улучшение результатов распознавания. Whisper (OpenAI)

Ниже пример скрипта `whisper_asr.py`, который использует модель Whisper (размер 139M)

```python

#!/usr/bin/env python3
# file: whisper_asr.py

import os
import whisper
from jiwer import wer, cer

model = whisper.load_model("base")  # можно взять tiny, small, medium, large
AUDIO_DIR = "/home/jupyter/lab_asr/data/audio"
TRANS_DIR = "/home/jupyter/lab_asr/data/transcripts"

refs, hyps = [], []

for fn in sorted(os.listdir(AUDIO_DIR)):
    if not fn.endswith(".wav"): continue
    utt = os.path.splitext(fn)[0]
    result = model.transcribe(os.path.join(AUDIO_DIR, fn), language="en", fp16=False)
    hyps.append(result["text"].lower().strip())
    refs.append(open(os.path.join(TRANS_DIR, f"{utt}.txt"), encoding="utf-8").read().lower().strip())
    print(f"{utt}  |  HYP: {result['text']}")

print("WER:", wer(refs, hyps))
print("CER:", cer(refs, hyps))

```

Запуск:

```bash

python whisper_asr.py

```

Результат работы:
```bash

WER: 0.22033898305084745
CER: 0.06396255850234009
```

Ниже представлена сравнительная таблица:
|               | Wav2Vec2 | Whisper (OpenAI) |
|---------------|----------------------------|-----------------------------|
| Объем модели | 1.26G | 139M |
| WER | 0.017 | 0.220 |
| СER | 0.006 | 0.064 |

---
### 📌 Задание для самостоятельной работы

Требуется:
1. Дополнить конвейер этапом фильтрации шума (например, библиотека librosa).
2. Сравнить метрики на чистом и зашумлённом аудио.
3. Провести анализ влияния длины фрагментов (короткие vs длинные) на WER/CER.

---

## 💡 Не забудьте выключить текущую среду выполнения программы python (должна пропасть надпись (venv) в начале командной строки):

```bash

deactivate

```


## Вопросы
1. Как изменение уровня шума влияет на WER и CER?
2. Какая метрика более чувствительна к разным видам ошибок – WER или CER?
3. Насколько различаются результаты при разных длинах аудиофрагментов?
4. Какие шаги предобработки дали наибольший выигрыш по качеству?
5. Какие ограничения первого подхода вы видите и как их можно преодолеть?



