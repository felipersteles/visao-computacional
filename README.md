# Visao Computacional

## How to use

### Before start

- First you must download the model in this link of google [drive](https://drive.google.com/drive/folders/1b1NAXEL4RzOI6kRLKG2i2qZrsQ7kbubx?usp=sharing). And then create a folder called `models/` and put the file the folder.

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

### Train()
