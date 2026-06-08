# Aneurysm-prediction-service
## Requerimientos
- [Docker](https://www.docker.com/)
- Container Image processing Backend levantado
- 1 o mas GPUS nvidia
- pro lo menos 16 gb de memoria libres
- los siguinetes deep learning models:
  - [VESSEL ROI SEGGMENTATION](https://www.kaggle.com/models/pengchengshi/dataset180_2d_vessel_box_seg_stable)
  - [ANEURYSM CLS 1 Y 2](https://www.kaggle.com/models/pengchengshi/rsna2025-stage2-models)
  - [PLANE 2D CLS](https://drive.google.com/drive/folders/1puJUTLiNyoqLPx3gLVY0yFrcjj493vDV?usp=sharing)
## Ejecuccion
```
docker compose -f docker-compose.yml up --build
```

## Tests:
Una vez levantado el container de docker se puede hacer
```
docker exec -it aneurysm-api /bin/sh -c "pip install pytest pytest-asyncio pytest-cov && python -m pytest -vv ./main_test.py && python -m pytest --cov=main /app/main_test.py --cov-report=term-missing"
```
