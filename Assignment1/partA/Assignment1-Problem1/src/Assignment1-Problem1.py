# coding: utf-8

# In[1]:


import configparser
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import unicodedata
from bs4 import BeautifulSoup, SoupStrainer, Comment
import csv
import pandas as pd
import logging
import os
import zipfile
import boto3
from botocore.client import Config
import argparse


def get_logger() :
    # create logger
    logger = logging.getLogger('Assignment-1-Problem1')
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    hdlr = logging.FileHandler('assignment-1-problem-1.log', mode='w')
    hdlr.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    hdlr.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(hdlr)

    return logger

log = get_logger()

def generateURL(baseUrl, cik, acc_no):

    cik_withoutZero = cik.lstrip('0')
    cik_accNo = acc_no.replace('-', '')
    url = baseUrl + cik_withoutZero + '/' + cik_accNo + '/' + acc_no + '-index.htm'
    log.info("Generated URL for EDGAR Data Set successfully")
    return url



def validUrl(url):
    try:
        http = urllib3.PoolManager()
        response = http.request('GET', url)
        print(response.status)
        log.info("Response Ok")
        return 'Valid'
    except Exception:
        print(Exception)
        log.error("Response not ok")
        return 'Not Valid'

def getResponse(url):
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    log.info("Response generated successfully")
    return response

def generate10QURL(response):
    soup = BeautifulSoup(response.data, "html.parser")
    table = soup.find('table', attrs={'summary': 'Document Format Files'})
    url_temp = table.find("td", text="10-Q").find_next_sibling("td").findChildren()
    for link in url_temp:
        url = link.get('href')

    url10q = 'https://www.sec.gov' + url
    log.info("Generated URL for 10-Q successfully")
    return url10q

def getTableList(response):
    soup = BeautifulSoup(response.data, "html.parser")
    tables = soup.find_all('table')
    filtered_table = list()

    for t in tables:
        count = 0
        for tr in t.find_all('tr'):
            for td in tr.find_all('td', style=True):
                if (td['style'].startswith('background:#B9CDE5')):
                    count = count + 1
                    break

        if count > 6:
            filtered_table.append(t)

    log.info("Scraped relevant tables")
    return filtered_table


def writeToCSV(filtered_table):
    count = 1
    tableList = list()
    os.makedirs(os.path.join('out-data'))
    for t in filtered_table:
        temp_table = list()

        with open(os.path.join("out-data/", str(count) + ".csv"), "w") as outcsv:
            writer = csv.writer(outcsv, delimiter=',', lineterminator='\n')

            for trows in t.find_all('tr'):
                row_arr = list()
                for tdesc in trows.find_all('td'):
                    row = tdesc.find_all('p')
                    for r in row:
                        if (r != ""):
                            row_arr.append(r.text.replace(u'\xa0', u' '))
                temp_table.append(row_arr)
                writer.writerow(row_arr)
            log.info("Writing into " + str(count) + ".csv")


            count = count + 1
            tableList.append(temp_table)
    zipf = zipfile.ZipFile(os.path.join('out-data.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(os.path.join('out-data'), zipf)
    zipf.close()
    log.info("Zipped the folder successfully")



def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))




def executeEdgarDataWrangling(cik, acc_no):

    baseUrl = 'https://www.sec.gov/Archives/edgar/data/'

    # cik = '0000051143'
    # acc_no = '0000051143-13-000007'

    url = generateURL(baseUrl, cik, acc_no)

    status = validUrl(url)

    if status == 'Invalid':
        log.error("Invalid URL")
    # break code here

    response = getResponse(url)

    url10q = generate10QURL(response)

    status = validUrl(url10q)

    if status == 'Invalid':
        print('Check URL')

    response = getResponse(url10q)

    filtered_table = getTableList(response)

    writeToCSV(filtered_table)

def upload_to_s3(public_id, private_id, bucket_name) :
    data_zip = open(os.path.join('out-data.zip'), 'rb')
    data_log = open(os.path.join('assignment-1-problem-1.log'), 'rb')
    s3 = boto3.resource(
        's3',
        aws_access_key_id=public_id,
        aws_secret_access_key=private_id,
        config=Config(signature_version='s3v4')
    )

    log.info("Public key/Private key Valid")
    # creating Bucket
    #s3.create_bucket(Bucket=bucket_name)
    # Image Uploaded
    s3.Bucket(bucket_name).put_object(Key='out-data.zip', Body=data_zip, ACL='public-read-write')
    s3.Bucket(bucket_name).put_object(Key='assignment-1-problem-1.log', Body=data_log, ACL='public-read-write')
    print('Uploaded to S3!')


def arg_parser():
    parser = argparse.ArgumentParser('CaseStudy1Part1')
    parser.add_argument('-c', '--cik', help='The CIK of the company.', type=str)
    parser.add_argument('-a', '--accno', help='The accession number of the company', type=str)
    parser.add_argument('-k', '--awskey', help='The key to the aws account where the files is to be uploaded.', type=str)
    parser.add_argument('-s', '--awssecretkey', help='The secret key to the aws account where the files is to be uploaded', type=str)
    parser.add_argument('-b', '--awsbucket', help='The bucket where the files is to be uploaded', type=str)
    args = parser.parse_args()

    if(args.cik != None) & (args.accno != None) & (args.awskey != None) & (args.awssecretkey != None) & (args.awsbucket != None):
      log.info("CIK, AccNo, AWS key pass and bucket valid")
    else:
        return 0

    return args


if __name__ == '__main__':

    try:
     args = arg_parser()
     cik = args.cik
     acc_no = args.accno
     ACCESS_KEY_ID = args.awskey
     ACCESS_SECRET_KEY = args.awssecretkey
     BUCKET_NAME = args.awsbucket

     executeEdgarDataWrangling(cik, acc_no)
     upload_to_s3(ACCESS_KEY_ID, ACCESS_SECRET_KEY, BUCKET_NAME)
    except Exception:
        log.error("Missing parameters: Either CIK, AccNo, AccessKey, SecretKey or BucketName missing")

