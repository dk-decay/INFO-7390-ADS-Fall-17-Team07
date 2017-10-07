FROM python:3

# Update
#RUN apk add --update python py-pip

# Install app dependencies
RUN pip install urllib3

RUN pip install requests

RUN pip install bs4

RUN pip install configparser

RUN pip install numpy

RUN pip install pandas

RUN pip install boto3

RUN pip install argparse

RUN pip install seaborn

RUN pip install matplotlib



# Bundle app source
COPY test.py /src/test.py

CMD ["python", "/src/test.py"]