# SexEst web application

This is the source code for the SexEst web application.
The live version of the application can be found [here](http://sexest.cyi.ac.cy/).

The source code runs inside a docker container which is assumed to 
be running either on your local machine or a web-server.
It is assumed that Docker is installed on your machine.
Instructions on how to install Docker on a *Mac* can be found 
[here](https://docs.docker.com/desktop/mac/install/) and for 
*Ubuntu* [here](https://docs.docker.com/engine/install/ubuntu/).
Alternatively for *Ubuntu* you may try:

`sudo apt-get install docker.io`



# Building the Docker Image

Once docker is setup on your machine you may clone this repository
to your local directory. You can then build the Docker image by running:

`docker build -f Dockerfile -t app:latest .`

# Running the Docker Image as a container locally

Once the image has been built, you can run a local instance of the *SexEst*
web application by running:

`docker run -p 8501:8501 app:latest`

If everything works, you can visit your web-app using the address
`http://localhost:8501/` in your browser.

# cleanup

docker rm -vf $(docker ps -aq)
docker rmi -f $(docker images -aq)