FROM python:3.7.1
LABEL NYC Taxi Fare Estimator, Brett Barnes, Chahn Ro, Northwestern University
COPY ./random-forest-model /usr/local/python/random-forest-model/
COPY ./api/ /usr/local/python/api/
EXPOSE 8080
WORKDIR /usr/local/python/api/
COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt
WORKDIR /usr/local/python/api/
CMD python Taxi_Fare_Flask.py