# Sentiment Analyser

Following the Udemy course: [A to Z (NLP) Machine Learning Model Building and Deployment](https://www.udemy.com/course/a-to-z-nlp-machine-learning-model-building-and-deployment/)

## Requirements

Docker, and / or Python3

## Instructions

- Spin up the container: `docker-compose up -d [--build]`
- (Open the container shell: `docker-compose exec app sh`)
- Open the app at `0.0.0.0:5000`
- Shut down the container: `docker-compose down`

### Local

- Install the dependencies: `pipenv install`
- Open the virtual env: `pipenv shell`
- Run the app: `python manage.py`
- Open the app at `0.0.0.0:4000`
- Exit the virtual env: `exit`

## Development

**Live reloading** is enabled. Any changes made to the code in the `project` directory and `requirements.txt` are immediately mounted in the running container.

(New) **dependencies** can be installed / updated with `docker-compose exec app pip install -r requirements.txt` in the Docker container, and `pipenv install <DEPENDENCY>` in the virtual env.

Run `black`, `flake8` and `isort` in the virtual env to **format** the code.

## Features

Enter a message to see if it is has a positive or negative connotation.

## Technologies

scikit-learn - Flask - Docker
