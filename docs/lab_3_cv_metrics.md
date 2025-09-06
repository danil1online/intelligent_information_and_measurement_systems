# Методические указания к лабораторной работе №3 
**Тема:** Метрики оценки качества генерации изображений  

---

## 🎯 Цель работы  
Изучить метрики оценки качества решения задачи «Генерация изображений» на практике.

---

## 📌 Задачи  
- Изучить принципы работы метрики Inception Score (IS). 
- Изучить принципы работы метрики Fréchet Inception Distance (FID).

---

## 📁 Материалы и методы
- Язык программирования – Python 3.10.
- Основные библиотеки: TensorFlow/Keras, scikit-learn, matplotlib, numpy, pillow, scipy
- Датасет: [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset):
1. 60 000 обучающих и 10 000 тестовых изображений размером 28×28
2. Грейскейл, один канал
3. Удобен для быстрой тренировки VAE и начальной оценки метрик

---

## 📚 Краткая теоретическая информация  

### 📚 Задача «Генерация изображений»

Генерация изображений подразумевает обучение модели, способной создавать новые изображения, схожие по распределению с обучающим набором. Часто используются архитектуры автокодировщиков (VAE), генеративных состязательных сетей (GAN) или диффузионных моделей. Качество работы таких моделей оценивают не только визуально, но и с помощью численных метрик.

### 📚 Inception Score (IS)

Inception Score основан на предсказаниях сети Inception v3 на сгенерированных образцах. Высокий IS достигается, если предсказания сети однозначны (низкая энтропия условного распределения) и разнообразны (высокая энтропия среднего распределения).

Формула IS:

$$
  IS = exp(E_{x} \cdot D_{KL} \cdot (p(y|x) \mathbin{||} p(y))
$$

### 📚 Fréchet Inception Distance (FID)

FID оценивает расстояние между многомерными гауссовыми распределениями признаков реальных и сгенерированных изображений, извлечённых сетью Inception v3. Чем меньше FID, тем ближе распределения.

Формула FID:

$$
  \text{FID} = \Vert \mu_r - \mu_g \Vert^2 + \mathrm{Tr} \left( C_r + C_g - 2 \cdot \left( C_r C_g \right)^{1/2} \right)
$$

## ⚙️ Настройка среды

0. Подключитесь к [Jupyter-Hub-ИИСТ-НПИ](http://195.133.13.56:8000/) из [первой работы](docs/lab_1_cv_metrics.md#%EF%B8%8F-настройка-среды)
1. Создайте в корне домашнего каталога каталог проекта и перейдите в него:
```bash

mkdir image_gen_lab
cd image_gen_lab

```
2. Создайте и активируйте виртуальное окружение:
```bash

python3.10 -m venv venv
source venv/bin/activate

```

3. Установите зависимости:
```bash

pip install --upgrade pip
pip install tensorflow-cpu scikit-learn matplotlib numpy pillow scipy imageio
```

---

## 🧪 Пример - метод VAE

### 🧪 Генерация изображений простым VAE на MNIST

Первым этапом примера является скрипт `vae_mnist.py` создания модели, обучения на mnist в течение 10 эпох и генерации изображений. Сгенерированные изображения затем будем оценивать с точки зрения качества 

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
import tensorflow as tf

def build_encoder(latent_dim=2):
    inputs = layers.Input(shape=(28,28,1))
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    z_mean    = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    def sampling(args):
        mean, log_var = args
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

    z = layers.Lambda(sampling)([z_mean, z_log_var])
    return Model(inputs, [z_mean, z_log_var, z], name='encoder')

def build_decoder(latent_dim=2):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(latent_inputs)
    x = layers.Dense(28*28, activation='sigmoid')(x)
    outputs = layers.Reshape((28,28,1))(x)
    return Model(latent_inputs, outputs, name='decoder')

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker        = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker  = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker     = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.loss_tracker,
                self.recon_loss_tracker,
                self.kl_loss_tracker]

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction      = self.decoder(z)

            x_flat   = tf.reshape(data,        [-1, 28*28])
            rec_flat = tf.reshape(reconstruction, [-1, 28*28])

            recon_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(x_flat, rec_flat)
            ) * 28 * 28

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var
                              - tf.square(z_mean)
                              - tf.exp(z_log_var),
                              axis=1)
            )

            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss":       self.loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss":    self.kl_loss_tracker.result(),
        }

def main(epochs=5, output_dir='output'):
    # 1. Загрузка и подготовка данных
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_train = np.expand_dims(x_train, -1)
    #vae, encoder, decoder = build_vae()
    # 2. Построение модели
    latent_dim = 2
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    vae     = VAE(encoder, decoder)
    #vae.compile(optimizer='adam')
    # 3. Обучение
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    
    #vae.fit(x_train, x_train, epochs=epochs, batch_size=128)
    vae.fit(x_train,
            epochs=10,
            batch_size=128)
    # 4. Сохранение decoder'а сразу после обучения
    os.makedirs(output_dir, exist_ok=True)
    decoder.save(os.path.join(output_dir, 'decoder.h5'))
    print(f"Decoder saved to {output_dir}/decoder.h5")
    # 5. Генерация и сохранение примеров
    z_sample = np.random.normal(size=(16, 2))
    x_decoded = decoder.predict(z_sample)
    plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(x_decoded[i].reshape(28,28), cmap='gray')
        plt.axis('off')
    plt.savefig(f'{output_dir}/vae_mnist.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='output')
    args = parser.parse_args()
    main(args.epochs, args.output_dir)

```

Запуск из командной строки:

```bash
python vae_mnist.py --epochs 10 --output-dir images
```

После запуска в каталоге images будет создан один файл `vae_mnist.png` - пример изображения и еще один файл - модель (`decoder.h5`), которая это изображение сгенерировала. 

### 🧪 Создание множества изображений с использованнием сохраненной модели

Для того, чтобы можно было объективно посчитать качество генерации одного изображения недостаточно - необходима т.н. репрезентативная выборка. 

Для ее создания используем следующий скрипт `generate_samples.py`

```python

import os
import numpy as np
from tensorflow.keras.models import load_model
import imageio

def main(decoder_path="images/decoder.h5",
         output_dir="images",
         n_samples=2000,
         latent_dim=2):
    os.makedirs(output_dir, exist_ok=True)
    decoder = load_model(decoder_path)

    # Берём N точек из N(0,1)
    z = np.random.normal(size=(n_samples, latent_dim))
    imgs = decoder.predict(z)  # shape (n_samples, 28,28,1)

    # Сохраняем
    for i, img in enumerate(imgs):
        # Масштабируем 0–1 → 0–255 и приводим к uint8
        arr = (img[:, :, 0] * 255).astype(np.uint8)
        imageio.imwrite(f"{output_dir}/sample_{i:05d}.png", arr)

    print(f"Saved {n_samples} images to {output_dir}")

if __name__ == "__main__":
    main()

```

Запуск скрипта максимально прост:

```bash

python generate_samples.py

```

В результате, в папке images должны появиться 2000 изображений, что можно считать достаточной выборкой для качественной оценки моделей компьютерного зрения.

### 🧪 Размещение датасета mnist в локальной папке

Скрип, который ранее обучил модель, работал с датасетом mnist, размещенным в ОЗУ. Для расчета метрик необходимо иметь такой датасет, размещенный в одном из каталогов, например, `data/mnist_real`

```python

# prepare_real_mnist.py
import os
import imageio
import numpy as np
from tensorflow.keras.datasets import mnist

def main(output_dir="data/mnist_real"):
    (x_train, _), _ = mnist.load_data()
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем первые 60 000 изображений
    for idx, img in enumerate(x_train):
        # MNIST хранится в 0–255, uint8
        imageio.imwrite(os.path.join(output_dir, f"{idx:05d}.png"), img)

    print(f"Saved {len(x_train)} images to {output_dir}")

if __name__ == "__main__":
    main()

```

Запуск скрипта также максимально прост:

```bash

python prepare_real_mnist.py

```

### 🧪 Вычисление IS и FID для сгенерированных изображений

И теперь мы готовы к расчету метрик - используем следующий скрипт  `eval_metrics.py`

```python

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
from PIL import Image
import glob

def load_images(path, size=(299,299), max_images=1000):
    files = glob.glob(os.path.join(path, '*.png'))[:max_images]
    imgs = []
    for f in files:
        img = Image.open(f).convert('RGB').resize(size)
        imgs.append(np.array(img))
    return np.array(imgs)

def calculate_inception_score(images, splits=10):
    model = InceptionV3(include_top=True, weights='imagenet', pooling='avg')
    images = preprocess_input(images.astype('float32'))
    preds = model.predict(images)
    p_y = np.mean(preds, axis=0)
    scores = []
    N = preds.shape[0] // splits
    for i in range(splits):
        part = preds[i*N:(i+1)*N]
        kl = part * (np.log(part+1e-10) - np.log(p_y+1e-10))
        scores.append(np.exp(np.sum(kl, axis=1).mean()))
    return float(np.mean(scores)), float(np.std(scores))

def calculate_fid(real, gen):
    model = InceptionV3(include_top=False, pooling='avg')
    act1 = model.predict(preprocess_input(real.astype('float32')))
    act2 = model.predict(preprocess_input(gen.astype('float32')))
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2*covmean)

def main(real_dir, gen_dir):
    real = load_images(real_dir)
    gen = load_images(gen_dir)
    is_mean, is_std = calculate_inception_score(gen)
    fid_value = calculate_fid(real, gen)
    print(f'Inception Score: {is_mean:.3f} ± {is_std:.3f}')
    print(f'FID: {fid_value:.3f}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', type=str, required=True)
    parser.add_argument('--gen', type=str, required=True)
    args = parser.parse_args()
    main(args.real, args.gen)


```

Запуск данного скрипта:

```bash

python eval_metrics.py --real data/mnist_real --gen images

```

---

## 📌 Задания для самостоятельной работы

1. Реализовать генерацию изображений на основе простой GAN и оценить её IS и FID.
2. Протестировать влияние размера латентного вектора на качество (сравнить IS/FID при разных размерностях).
3. Провести сравнение метрик для MNIST и поднаборов CIFAR-10.

---

## 💡 Не забудьте выключить текущую среду выполнения программы python (должна пропасть надпись (venv) в начале командной строки):

```bash

deactivate

```


## Вопросы
1. Как Inception Score отражает разнообразие и качество сгенерированных изображений?
2. В каких ситуациях IS может быть необъективной метрикой?
3. Почему FID считается более надёжным, и какие его ограничения?
4. Как влияет размер выборки на стабильность FID?
5. Какие архитектурные особенности модели генерации сильнее всего влияют на значения IS и FID?
