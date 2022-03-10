# Flasgger API

docker build -t sexest .

docker run -p 8000:8000 sexest

# streamlit app

docker build -f Dockerfile -t app:latest .

docker run -p 8501:8501 app:latest &

# cleanup

docker rm -vf $(docker ps -aq)
docker rmi -f $(docker images -aq)