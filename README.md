# Hands on with Background Matting library

Hi! This is my hands on experience with an interesting library for performing background replacement of images or videos.

Link to the Github repository: https://github.com/PeterL1n/BackgroundMattingV2

Link to paper: https://arxiv.org/abs/2012.07810

![alt text](https://github.com/DarrelYee/Background_Matting_Tutorial/blob/main/demo.png)

Exisiting background removal methods tend to use a binary mask to separate the foreground subject from the background; see above image for an example of Zoom's implementation (middle example). This method of masking produces a result that contains artifacts near finely-detailed portion of the image, including hair or semi-transparent objects. The alternative is to use chroma key compositing with a fixed-colour background (also known as green screening) as commonly performed in news broadcoasts and CGI.

Lin et al. proposes a new method that is able to achieve much higher quality result with respect to fine details, while also running in real time, without the need for a fixed-colour background. To achieve this, the model uses an additional input to the model: image of the pure background, and outputs an alpha-level masking (known as alpha matte in the paper) of the subject foreground. In essence, this is a segmentation model which outputs an alpha mask instead of a binary mask, and is able to do so in real time.

![alt text](https://github.com/DarrelYee/Background_Matting_Tutorial/blob/main/model.png)

At a high level, the model first uses a base encoder-decoder network to form a basic alpha mask, as well as an error map which is used for further fine-tuning. In the refinement network, regions proposed by the error map are used to provide high-resolution matting details, usually to the edges of the subject. The output of this model is the final alpha mask and foreground residual, which can then be composited to obtain the subject foreground. The model is trained on VideoMatte240K and PhotoMatte13K/85 datasets with alpha masks created from chroma key methods.

More can be read from their paper (link above!), but for now let's move on to some implementation!

## Implementation

For this hands-on I will be running my code off Google Colab as all the dependencies are already installed. The code is adapted from resources on the author's Github page (link above!), but edited for a step-by-step explanation. While I explain the individual steps here, you can refer to my jupyter notebook in this repo for the full code.

First, import the following libraries:

```python
import os
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from google.colab.patches import cv2_imshow
from PIL import Image
import matplotlib.pyplot as plt
import cv2
```

Then, clone or copy the Github repo contents to your Gdrive and make it your working directory. In our code we manually downloaded the repo and linked to it with os.chdir():
```python
from google.colab import drive
drive.mount('/content/gdrive')

# Change this to your repo folder.
os.chdir('//content//gdrive//My Drive//BackgroundMattingV2-master')
```

The author has provided support for several different deep learning frameworks, but we shall be using the Pytorch version for this tutorial. Import the necesary scripts, and load the model weights as shown below:
```python
from model import MattingRefine

device = torch.device('cuda')
precision = torch.float32

# Change the backbone arg to whichever model you downloaded
model = MattingRefine(backbone='mobilenetv2',
                      backbone_scale=0.25,
                      refine_mode='sampling',
                      refine_sample_pixels=80_000)

# Replace this path with that of your downloaded weights.
model.load_state_dict(torch.load('pytorch_mobilenetv2.pth'))
model = model.eval().to(precision).to(device)
```
Before continuing, ensure you have downloaded the weights for the model (refer to Github page for links). Note that the weights we are using are from the mobilenetV2 dataset. This is the most lightweight of the three datasets the authors used for training, which also includes ResNet-50 and ResNet-101. For our purposes mobilenetV2 will suffice.

The next step is to run our model! Load your source and background images and convert them to tensors as shown below. The repo contains links to several test images and backgrounds to try out. Once everything is setup, we can run the model:

```python
# Load source & background images, and convert them to Pytorch tensors.
src = Image.open('img.png')
bgr = Image.open('bgr.png')
src = to_tensor(src).cuda().unsqueeze(0)
bgr = to_tensor(bgr).cuda().unsqueeze(0)

# Run model on the inputs.
with torch.no_grad():
    pha, fgr = model(src, bgr)[:2]
```

## Results

Below is a figure showing our source image (TL), background (TR), output alpha mask (BL) and foreground residual (BR).

![alt text](https://github.com/DarrelYee/Background_Matting_Tutorial/blob/main/comparison.png)

By multiplying the alpha mask to the residual, we can obtain the masked foreground. By inverting the alpha mask, we can also apply it to a new test background which we want to apply the foreground to.

```python
# Load replacement background image.
new_bgr = np.array(Image.open('beach.jpg'))

# Convert model outputs to numpy arrays
pha_arr = np.array(to_pil_image((pha)[0]))
fgr_arr = np.array(to_pil_image((fgr)[0]))

# Form the inverse alpha mask
inv_pha = 255 - pha_arr

# Alpha-mask the foreground.
fg = fgr_arr.astype('float32')
fg[:,:,0] = fg[:,:,0] * pha_arr / 255
fg[:,:,1] = fg[:,:,1] * pha_arr / 255
fg[:,:,2] = fg[:,:,2] * pha_arr / 255
fg = fg.astype('uint8')

# Inverse-alpha-mask the new background.
bg = new_bgr.astype('float32')
bg[:,:,0] = bg[:,:,0] * inv_pha / 255
bg[:,:,1] = bg[:,:,1] * inv_pha / 255
bg[:,:,2] = bg[:,:,2] * inv_pha / 255
bg = bg.astype('uint8')
```

The image below shows the masked foreground (top) and inverse-masked new background (bottom).

![alt text](https://github.com/DarrelYee/Background_Matting_Tutorial/blob/main/bgfg.png)

After that, all that is required is to add the two images together, and the following will result:

```python
# Add both to form the composite image
composite = bg + fg
```

![alt text](https://github.com/DarrelYee/Background_Matting_Tutorial/blob/main/result.png)

Notice that fine details such as hair are well-contrasted with the background without being clumped together. In theory, with this model anything background can now be removed, leaving only the subjects. The model is currently only trained on human images, therefore it can only discern human foreground subjects.

This tutorial thus covers the application of the model to single images. However, this can easily be applied to video input by performing a frame read and applying the model to each frame. Real-time video application of this model will require a parallel processing architecture which is beyond the scope of this tutorial.

Thanks for reading!


_Rights to all images belong to their respective authors._
