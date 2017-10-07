docker build -f Dockerfile .


docker tag <tag_name> devesh24/problemb:latest


docker push devesh24/problemb



docker run devesh24/problemb python /src/test.py -y <year> -k <public> -s <private> -b <existing_bucket_name>
