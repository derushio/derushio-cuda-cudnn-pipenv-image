from keras.applications.resnet50 import ResNet50
from keras.datasets import cifar10

from PIL import Image
import numpy as np

import json

## モデル読み込み
model = ResNet50(include_top=True, weights='imagenet',
    input_tensor=None, input_shape=None, pooling=None, classes=1000)

## ラベル読み込み
with open('image-net-labels.json') as f:
    labels = json.load(f)

## データ読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

## データサンプリング
x_test = x_test[:10]

## 推論
results = []
for i, array in enumerate(x_test):
    ## 画像読み込み
    image = Image.fromarray(array)
    ## リサイズ (input tensorのshapeに合わせる)
    image = image.resize((224, 224))
    ## tensor化
    y = np.asarray(image)
    ## 次元拡張 (input tensorのshapeに合わせる)
    y = np.expand_dims(y, axis=0)

    ## 推論
    y_probs = model.predict(y)
    ## 推論結果で一番可能性が高いものを取得
    y_class = y_probs.argmax(axis=-1)

    ## build result
    result = {}
    result['class'] = labels[str(y_class[0])]
    result['probability'] = y_probs[0][y_class][0]
    results.append(result)

print(results)
