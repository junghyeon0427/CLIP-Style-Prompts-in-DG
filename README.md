# KCC2024
Domain Generalization in Semantic Segmentation using CLIP Style Prompts   
**Junghyeon Seo, Sungsik Kim, Seungheon Song, Jaekoo Lee, Korea Computer Congress 2024 (KCC 2024)**  

## Method  
![image](https://github.com/junghyeon0427/KCC2024/assets/77001598/e47575dd-9369-42ec-86bf-45c79f68397c)

## Experimental Results
| Method                          | Cityscapes | Night  | Rain   | Snow   | Fog    |
|---------------------------------|------------|--------|--------|--------|--------|
| PØDA (Cityscapes → Night)       | 63.85      | **24.58** | 41.54 | 44.34 | 50.93  |
| PØDA (Cityscapes → Rain)        | 63.87      | 22.92  | **40.78** | 43.49 | 50.41  |
| PØDA (Cityscapes → Snow)        | 64.31      | 21.38  | 40.21  | **43.69** | 50.43  |
| PØDA (Cityscapes → Fog)         | 63.25      | 22.44  | 38.04  | 41.31  | **48.44** |
| **Ours (DG)**                   | **64.78**  | 23.26  | **43.60** | **44.69** | **52.44** |
