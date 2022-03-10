# SexEst web application

This is the source code for the SexEst web application.
The live version of the application can be found [here](http://sexest.cyi.ac.cy/).

# streamlit app

docker build -f Dockerfile -t app:latest .

docker run -p 8501:8501 app:latest &

# cleanup

docker rm -vf $(docker ps -aq)
docker rmi -f $(docker images -aq)