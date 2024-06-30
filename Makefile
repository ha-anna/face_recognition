init:
	pip install -r requirements.txt

run:
	python3 './src/face_recognition/main.py'

up:
	pip freeze > requirements.txt

format:
	black './src/face_recognition'
	black './tests'

test:
	python3 -m pytest .

check:
	ruff check

fix:
	ruff check --fix