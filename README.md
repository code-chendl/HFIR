# HFIR (Hybrid Fusion Based on Information Relevance)
This is the implementation of our paper "Joint Multimodal Sentiment Analysis Based on Information Relevance", including the source code and manually annotated datasets.
# Requirements
We give the version of the python package we used, please refer to `envs\versions.txt`.  
Or create an environment using `envs\environment.yaml` with conda.  

# Data
- **MSA-IR dataset:** For the protection of copyright, we cannot provide the origin tweets in MSA-IR. Instead, we provide the preprocessed data in the form of pickles. Available at:  

百度网盘：
 ```
Link：https://pan.baidu.com/s/1iG-1EawR7q9Qi0XMTGAidw?pwd=2022
Password：2022
```

Google drive:
```
Link：https://drive.google.com/drive/folders/1blLhut17mgQrViA-EWm6xKkMrAs2UxZQ?usp=sharing
```

  
- **Twitter-15/17 dataset:** We provide text data and annotations in `Twitter-1517\`.   
                             As for images, please download from the link below: 

百度网盘：
 ```
Link：https://pan.baidu.com/s/1GP5Ysu_-C-gg_aX5T_j8Ng?pwd=2022 
Password：2022
```

Google drive:
```
Link: https://drive.google.com/drive/folders/1hsHGGgQCw8w0CVLaGqkWTCxcDXVBFcA5?usp=sharing
Password：2022
```
# Instructions for manual annotation    
(1)	***Txt_label*** represents the sentiment polarity conveyed by textual description. Each text is given a label from the set {negative (labeled as -1), neutral (0), positive (1)}.    
(2)	***Img_label*** represents the sentiment polarity conveyed by visual content. Each image is given a label from the set {negative (-1), neutral (0), positive (1)}.    
(3)	***Multi_label*** represents the sentiment polarity conveyed by the whole image-text post. Each multimodal tweet is given a label from the set {negative (-1), neutral (0), positive (1)}.       
(4)	***Cor_label*** represents the information relevance between an image and its corresponding text. Each multimodal tweet is given a label from the set {relevant (labeled as y), irrelevant (n)}.    
- Note that, because Web users may post tweets without the restriction on image-text correlation, text and image in a message are not necessarily assigned the same sentiment label.  

- ***One-hot encoding*** {positive: [1 0 0], neutral: [0 1 0], negative: [0 0 1]; relevant: [1 0], irrelevant: [0 1]}

# Code
- **DeepSentiBank:** We utilize DeepSentiBank, pre-trained over 800K annotated images, to extract the mid-level features and Adjective Noun Pairs (ANPs).     
                     Output from fc7 layer -> Mid-level visual representation;  
                     Adjective Noun Pairs (ANPs) -> High-level visual features (along with tags extracted by VGG16, VGG19, Xception, Inception and Resnet);    
                     Please download the source code and pre-trained model from the link below: https://www.ee.columbia.edu/ln/dvmm/vso/download/sentibank.html    
- **GloVe:** We use the pre-trained GloVe model (glove.twitter.27B.200d) to encode word vectors. Available at:    

百度网盘：
 ```
Link: https://pan.baidu.com/s/1CFqUbVFmY92BrVR8a-wbuw?pwd=2022     
Password：2022
```

Google drive:
```
Link: https://drive.google.com/drive/folders/1H7yRhHzTQ7upmR-5fYY3WaGIaJxtV3yv?usp=sharing
```

# Citation
If you find this code or data useful for your research, please consider citing:

```

@article{chen2023joint,
  title = {Joint multimodal sentiment analysis based on information relevance},
  author = {Chen, Danlei and Su, Wang and Wu, Peng and Hua, Bolin},
  journal = {Information Processing \& Management},
  volume = {60},
  number = {2},
  pages = {103193},
  year = {2023},
  publisher = {Elsevier},
  doi = {10.1016/j.ipm.2022.103193}
}
