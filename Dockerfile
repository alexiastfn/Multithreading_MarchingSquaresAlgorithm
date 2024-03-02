FROM jokeswar/base-ctl

RUN apt-get update -yqq && apt-get install -yqq bc

COPY ./checker ${CHECKER_DATA_DIRECTORY}