# -*- coding: utf-8 -*-
"""
Graphs wordclouds of collected reviews
Author: DanElias
Date: May 2021
"""

"""### Import python libraries"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

"""# Load Data"""

df = pd.read_csv("../data/kelloggs_reviews_unlabelled.csv", encoding="UTF-8", usecols=["text"])
#df.head()
#df.shape

"""# Word Cloud - Frequently used words"""

def random_color_func(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):
    h = int(360.0 * 246.0 / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(50, 150)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)

# Start with one review:
text = " ".join(review for review in df.text)

stopwords = set(STOPWORDS)
stopwords.update(["really", "make", "much", "will", "got", "tried", "br", "little", "one", "bit", "even"])

wordcloud = WordCloud(max_font_size=50, color_func=random_color_func, max_words=500, background_color="white", stopwords=stopwords).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

tony_img = Image.open("../assets/tony-tiger.png")
tony_mask = np.array(tony_img)

wordcloud = WordCloud(max_font_size=50, color_func=random_color_func, max_words=500, background_color="white", stopwords=stopwords, mask=tony_mask).generate(text)
plt.figure(figsize=[15,15])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

tony_img2 = Image.open("../assets/tony3.png")
tony_mask2 = np.array(tony_img2)

wordcloud_tiger = WordCloud(stopwords=stopwords, background_color="white", max_words=1000, mask=tony_mask2, mode="RGBA").generate(text)

image_colors = ImageColorGenerator(tony_mask2)
plt.figure(figsize=[16,16])
plt.imshow(wordcloud_tiger.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.savefig("../results/wordcloud-tony.png", format="png")