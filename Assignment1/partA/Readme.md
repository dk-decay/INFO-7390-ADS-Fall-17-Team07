docker build -f Dockerfile .


docker tag <tag_name> devesh24/problema:latest


docker push devesh24/problema


docker run devesh24/problema python /src/Assignment1-Problem1.py -c <cik> -a <accession_no> -k <public> -s <private> -b <bucket_name>
