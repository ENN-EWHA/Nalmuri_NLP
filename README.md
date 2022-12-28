# Nalmuri_NLP
글에서 감정을 분석해주는 모델 +  Flask RestAPI

```shell
pip install -r requirements.txt
python server.py
```
# 주의사항
- 로컬에서만 가능

- model은 크기가 커서 따로 올리지 않음. 아래 사진과 같이 model 폴더에 따로 추가해줘야 함.
- 가상환경 구축하여 사용할 것을 권장

![image](https://user-images.githubusercontent.com/87990290/209721942-0b16c1aa-bdbb-4d01-a3d4-aeee6e635e16.png)

# Endpoint
### 감정 분석
```python
POST  /api

# request sample
# {
#  "sentence" : "타코야끼를 먹었다. 정말 행복한 하루였다." 
# }

```
