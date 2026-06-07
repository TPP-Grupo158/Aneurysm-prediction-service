# Aneurysm-prediction-service
### Requerimientos
- [Docker](https://www.docker.com/)
- Container Image processing Backend levantado
- 1 o mas GPUS nvidia
- pro lo menos 16 gb de memoria libres
### Ejecuccion
```
docker compose -f docker-compose.yml up --build
```
### Tests:
Una vez levantado el container de docker se puede hacer
```
docker exec -it aneurysm-api /bin/sh -c "pip install pytest pytest-asyncio pytest-cov && python -m pytest -vv ./main_test.py && python -m pytest --cov=main /app/main_test.py --cov-report=term-missing"
```
