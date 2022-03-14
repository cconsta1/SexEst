# SexEst web application

This is the source code for the SexEst web application.
The live version of the application can be found [here](http://sexest.cyi.ac.cy/).

The source code runs inside 
a docker container on your local machine or a web-server.
Because the web application uses Docker, it is 
assumed that Docker is installed on your machine or the web server
that will host the application.
Instructions on how to install Docker on a *Mac* can be found 
[here](https://docs.docker.com/desktop/mac/install/) and for 
*Ubuntu* [here](https://docs.docker.com/engine/install/ubuntu/).
Additionally, still for *Ubuntu*, you may try using the following command
in your terminal:

`sudo apt-get install docker.io`



# Building the docker image

Once docker is setup on your machine, you may clone this repository
to your local directory. 

Once the repository is cloned, you can build the Docker image 
by running the command:

`docker build -f Dockerfile -t app:latest .`

# Running the docker image as a container locally

Once the image has been built, you can run a local instance of the *SexEst*
web application by running:

`docker run -p 8501:8501 app:latest`

If everything works, you can visit your web-app using the address
`http://localhost:8501/` in your browser.

# Running the docker image as a container on a web server

Assuming that you have already installed docker on your webserver 
(`sudo apt-get install docker.io` should work), you can build the 
image in the same fashion as you would build it locally:

`docker build -f Dockerfile -t app:latest .`

You may then run the image as a container on the webserver:

`docker run --restart always -p 80:8501 app:latest`

We use the flag `--restart` and we set it to `always` to ensure that
if the server goes down for whatever reason the image will be run
automatically as soon as the server is back up again. 

The web app will now be accessible from the IP address assigned
to your webserver.

# Cleanup

To check the status of your running container you may use the command:

`docker ps`

To stop a running container you can use the command:

`docker stop <CONTAINER ID>`,

where the `<CONTAINER ID>` can be found in the first column of the output
of `docker ps`.

To remove all containers and all images (stop the containers first)
you may use:

`docker rm -vf $(docker ps -aq)`

`docker rmi -f $(docker images -aq)`