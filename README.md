lokalna instalacja pakietu w trybie deweloperskim 
pip install -e .[dev]
# [dev] - extra dependencies

FAST API DOCs
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc

pytest -v

pip install -e .[dev]
pip install -e . 

Uruchom wszystkie testy:
pytest

Uruchom testy z raportem pokrycia kodu (jeśli masz pytest-cov):
pytest --cov=matrix_utils