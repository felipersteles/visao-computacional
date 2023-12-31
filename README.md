# Computer Vision

Ah, the age-old battle of Rock, Paper, Scissors. A timeless game of chance, skill, and maybe a little bit of luck. But what if we told you that soon, machines might be able to predict the outcome – not through psychic powers, but through the magic of artificial intelligence?

I developed a Classification Convolutional Neural Network and called `TelesNet`, which the use is to classify the gesture inserted in a game of rock, paper and scissors.

## Why Rock, Paper, Scissors?

Beyond its nostalgic charm, this classic game provides an ideal testing ground for AI development. With just three distinct hand gestures, it's a manageable problem with clear rules and outcomes. Yet, despite its apparent simplicity, it's not just about brute-force pattern recognition. It requires understanding subtle differences in hand shapes, lighting, and even player psychology.

## How to use

### Before start

- First you must download the model in this link of google [drive](https://drive.google.com/drive/folders/1b1NAXEL4RzOI6kRLKG2i2qZrsQ7kbubx?usp=sharing). And then create a folder called `models/` and put the file into the folder.

- Install the dependencies using:

```bash
# Install dependencies
$ pip install -r requirements.txt

```

> If you running in windows, maybe you'll have a problem to install scikit lib so i recommend you to use pipwin:

```bash
# Install pipwin
$ pip install pipwin
|----------------| 100%

$ pipwin install scikit-learn
|----------------| 100%

$ pipwin install scikit-image
|----------------| 100%

```

### Play()

After that you can use the method `play()` to open the web cam and make the model try to discover the hand gesture that could be _rock, paper or scissors_.

```python
from TelesNet import TelesNet

model = TelesNet()

model.play()

```

> Obs: The model was trained with images on a green background, so for the model to have a better chance of getting the gesture right, make sure the hand is appearing behind a clean environment, such as a white wall or a green screen.

When the method is running will be two commands:

- Close the window `(Press Q)`
- Classify the gesture `(Press C)`

![image](./imgs/demo.png)

### Train()

Before train the model, it is necessary to call the `setData` function to insert the training data. An example of data is a array with images and labels of each image in the same index.

> Example:

```python
# Load data
data, labels = load_data()

# Use the setTrain method
model.setData(data, labels, numClasses=2, classes=["cat","dog"])

# Choose a version
model.buildModel("v1")

# Train
model.train()

```
## Materials 
A dataset of labeled images showing hands forming rock, paper, and scissors gestures, and it's avaiable in this [link](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors).

## Models

Visualization of networks in an understandable way.

### V1

![v1](./imgs/v1.png)

### V2

![v2](./imgs/v2.png)

### V3

Transfer learning from EfficientNet

![v3](./imgs/v3.png)

## Training
The training was carried out with x images and therefore a batch size of 64 was chosen. In addition to the models, a DataAugmentation was also implemented with a focus on contrast so that the model generalizes well with a background of any color. The training lasted 50 epochs and the metrics chosen for evaluation were precision, recall and accuracy.

### Results

|   | Validation | Prediction |
|---|---|---|
| Precision | 0.9886 | 0.8901 |
| Accuracy| 0.9628 | 0.8630 |
| Recall | 0.9628 | 0.8630 |

## Conclusion

In this exploration, I built and trained a simple neural network to classify the classic game of Rock, Paper, Scissors. Using TensorFlow and image recognition, the model achieved an accuracy of 0.8630 on the test dataset. This result demonstrates the potential of neural networks for image classification tasks, even for seemingly simple applications like hand gestures.
