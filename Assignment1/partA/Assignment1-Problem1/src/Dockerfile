FROM python:3

# Update
#RUN apk add --update python py-pip

# Install app dependencies
RUN pip install urllib3

RUN pip install bs4

RUN pip install configparser

RUN pip install numpy

RUN pip install pandas

RUN pip install boto3

RUN pip install argparse

# Bundle app source
COPY Assignment1-Problem1.py /src/Assignment1-Problem1.py

CMD ["python", "/src/Assignment1-Problem1.py"]