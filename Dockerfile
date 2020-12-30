FROM python:3.6
WORKDIR /deploy/
COPY ./requirements.txt /deploy/
COPY ./webapp /deploy/webapp
COPY ./instance /deploy/instance
RUN pip install -r ./requirements.txt
CMD ["python", "webapp/__init__.py"]~