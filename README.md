# HFIR (Hybrid Fusion Based on Information Relevance)
This is the implementation of our IPMC2022 paper "Joint Multimodal Sentiment Analysis Based on Information Relevance", including the source code and manually annotated datasets.
# Requirements
We give the version of the python package we used, please refer to `envs\environment.yaml`
# Data
- **MSA-IR dataset:** For the protection of copyright, we cannot provide the origin tweets in MSA-IR. Instead, we provide the preprocessed data in the form of pickles. Available at:  

百度网盘：
 ```
Link：https://pan.baidu.com/s/1iG-1EawR7q9Qi0XMTGAidw?pwd=2022#list/path=%2F
Password：2022
```

Google drive:
```
https://drive.google.com/drive/folders/1blLhut17mgQrViA-EWm6xKkMrAs2UxZQ?usp=sharing
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

# Code
- **DeepSentiBank:** We utilize DeepSentiBank, pre-trained over 800K annotated images, to extract the mid-level features and Adjective Noun Pairs (ANPs).   
                     Output from fc7 layer -> Mid-level visual representation;  
                     Adjective Noun Pairs (ANPs) -> High-level visual features (along with tags extracted by VGG16, VGG19, Xception, Inception and Resnet);    
                     Please download the source code and pre-trained model from the link below: https://www.ee.columbia.edu/ln/dvmm/vso/download/sentibank.html  


# Citation
If you find this code or data useful for your research, please consider citing:

```

@inproceedings{chen2022HFIR,
    title = "Joint Multimodal Sentiment Analysis Based on Information Relevance",
    author = "Danlei Chen and Wang Su and Peng Wu and Bolin Hua",
    booktitle = "Information Processing & Management Conference 2022: IPMC 2022",
    year = "2022"
}
