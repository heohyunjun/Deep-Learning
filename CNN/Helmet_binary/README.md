# 헬멧 탐지를 위한 CNN

> Yes_Helmet , No_Helmet  이진 분류
> 

## 데이터셋

- Open Image Dataset
    - Helmet
    - Bicycle Helmet
    - Human head
    
- OS
    - Centos

- GPU
    - V100
    
- 전처리
    - RetinaFace를 사용해 얼굴 부분 크롭하여 사용 데이터셋 전처리
    - Helmet, Biycle Helmet 이미지에서는 y좌표값을 늘려서 헬멧 부분까지 크롭
    - 데이터 정제
    - 데이터 비율 맞추기
