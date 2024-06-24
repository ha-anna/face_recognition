run:
	python3 './src/face_recognition_ha-anna/main.py'

up:
	pip freeze > requirements.txt

format:
	black './src/face_recognition_ha-anna'

test:
	pytest .
