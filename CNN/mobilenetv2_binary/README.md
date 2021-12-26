# 제목 없음

# 헬멧 탐지를 위한 Mobilenetv2

> Yes_Helmet , No_Helmet  이진 분류
> 

## 데이터셋

- Open Image Dataset
    - Helmet
    - Bicycle Helmet
    - Human head

- 전처리
    - RetinaFace를 사용해 얼굴 부분 크롭하여 사용 데이터셋 전처리
    - Helmet, Biycle Helmet 이미지에서는 y좌표값을 늘려서 헬멧 부분까지 크롭

- 데이터 정제
    - 각 클래스에서 잘못 크롭된 이미지 삭제