FROM python:3
COPY app .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 1000
#CMD('python', 'main.py')