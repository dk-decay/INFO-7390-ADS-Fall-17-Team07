import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt


import zipfile
from multiprocessing import Pool
import requests
import pandas as pd
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import six
from datetime import datetime
import logging
import argparse
import boto3
from botocore.client import Config


# Import configuration file at the global Level, so that the scope of the config is Global.
# You can read as well as update the config file in this code.
# config = configParser.ConfigParser()
# config.read('../config/assignment1-problem2.ini')

# This is the worker process that will extract the zio file and return the dataframe.
# Any processing that is to be done to the individual files before appending their results can be done here.
# This will reduce amount of processing that needs to be done on the appended content
os.makedirs(os.path.join('output-summary'))
pdf = PdfPages(os.path.join('output-summary/summary.pdf'))


def get_logger() :
    # create logger
    logger = logging.getLogger('Assignment-1-Problem2')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    hdlr = logging.FileHandler('assignment-1-problem-2.log', mode='w')
    hdlr.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    hdlr.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(hdlr)

    # 'application' code
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical messa')
    return logger


def worker(procnum):
    print('Worker :', procnum)

    try:
        url_array = procnum.split('/')
        file_name = url_array[len(url_array) - 1]
        csv_file_name = file_name.split('.')[0] + '.csv'

        response = requests.get(procnum)
        zfile = open(os.path.join('temp/' + file_name), "wb")
        zfile.write(response.content)
        zfile.close()


        zip_file_object = zipfile.ZipFile(os.path.join('temp/' + file_name), 'r')
        zip_file_object.extract(csv_file_name, path = os.path.join('temp'))

        zip_file_object.close()
        os.remove(os.path.join('temp/' + file_name))
        return csv_file_name
    except Exception as e :
        print('file messed up :', e)


# This method initiates variables and passes it to the Pool method which enables multiprocessing.
# Pool.close and Pool.join ensure that the control is returned to the main process after the executing all the sub processes.
# This method returns the dataframe object from each subprocess
def enableMultiProcessing(year):

    try:
        print('enabled multi processing')


        base_url = 'http://www.sec.gov/dera/data/Public-EDGAR-log-file-data'
        date_quater_tuple = generate_all_iso_date(year)
        os.makedirs(os.path.join('temp'))
        url_array = [generate_url(base_url, quarter, year, iso_date) for (iso_date, quarter) in date_quater_tuple]

        # pool = Pool(12)
        # results = pool.map(worker, url_array)

        results = list()
        for url in url_array :
            results.append(worker(url))

        # pool.close()
        # pool.join()
        return results

    except Exception as e:

        print('Devesh Fucked up!!!' ,str(e))


def generate_url(base_url, quater, year, isoDate):
    return base_url + '/' + str(year) + '/' + quater + '/' + 'log' + isoDate + '.zip'

def generate_isoDate(year, month, date):

    return str(year) + month + date

def generate_quater(month):
    if month < 4 :
        return 'Qtr1'
    elif month < 7 :
        return 'Qtr2'
    elif month < 10 :
        return 'Qtr3'
    else :
        return 'Qtr4'


def generate_all_iso_date(input_year) :
    return [(generate_isoDate(input_year, "{0:0=2d}".format(month), '01'),
             generate_quater(month)) for month in range(1 ,13)]


def process_files(file_name_array):
    df = pd.DataFrame()
    for file_name in file_name_array :
        df = df.append(pd.read_csv(os.path.join('temp/' + file_name)), ignore_index= True)



    determine_nan_columns(df)
    df = merge_date_time_column(df)
    df = clean_missing_data(df)
    df = summary_metrics(df)
    os.makedirs(os.path.join('problem2-out-data'))
    write_output_data(df, os.path.join('problem2-out-data/new-data.csv'))

    zipf = zipfile.ZipFile(os.path.join('problem2-out-data.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(os.path.join('problem2-out-data'), zipf)
    zipf.close()

    print('Completed everything!!')




def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def write_output_data(df, name) :
    df.to_csv(name, sep=',', index = False)
    return

def determine_nan_columns(df) :
    #plt.figure(figsize=(5, 5))
    fig = plt.figure()
    nan_dist = df.isnull().mean(axis=0)
    nan_dist[nan_dist > 0].plot.barh()
    plt.title("'This table shows the average number of NaN values for each column as a bar graph'")


    pdf.savefig(fig)
    null_df = df.apply(lambda x: sum(x.isnull()), axis=0).reset_index()
    null_df.columns = ['Column_Name', 'No of NaNs']
    tab_fig = render_mpl_table(null_df, header_columns=0, col_width=6.0)

    pdf.savefig(tab_fig)


def render_mpl_table(data, col_width=10.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    fig = None
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return fig


def merge_date_time_column(df) :
    df['datetime'] = df['date'].map(str) + ' ' + df['time'].map(str)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.drop(['date', 'time'], axis=1, inplace=True)
    return df




def clean_missing_data(df) :

    filtered_df = df[['ip', 'browser', 'cik']]
    not_null_filtered_df = filtered_df[filtered_df['browser'].notnull()]
    de_deplicated_df = not_null_filtered_df.drop_duplicates()
    de_deplicated_df = de_deplicated_df.groupby(by=['ip', 'browser']).count()

    de_deplicated_df = de_deplicated_df.reset_index()
    de_deplicated_df = de_deplicated_df.sort_values(by=['cik'], ascending=False).groupby(by=['ip', 'browser']).head(1)

    de_deplicated_df = de_deplicated_df[['ip', 'browser']]

    new_df = pd.merge(df, de_deplicated_df, on='ip', how='outer')


    new_df['browser_y'].fillna('Unknown', inplace=True)
    new_df.drop(['browser_x'], axis=1, inplace=True)
    new_df.rename(columns={'browser_y': 'browser'}, inplace=True)

    extention_size_dict = new_df[['extention', 'size']].drop_duplicates()[new_df['size'].notnull()].groupby('extention').mean()

    new_df = new_df[new_df['ip'].notnull()]
    new_df = new_df[new_df['cik'].notnull()]
    new_df = new_df[new_df['accession'].notnull()]

    for index, row in new_df.iterrows():
        if row['size'] == np.NaN:
            row_extention = row['extention']
            row['size'] = extention_size_dict.loc[row_extention]

    return new_df


def summary_metrics(df):
    df['day_of_week'] = df['datetime'].dt.weekday_name
    weekday_vs_weekend_analysis(df)
    df['quarter'] = df['datetime'].dt.quarter
    top_cik_per_quarter(df)
    df['month'] = df['datetime'].dt.month
    top_3_ip_per_month(df)
    get_anomaly_top_100(df)
    top_cik_file_size_anomaly(df)
    df['hour'] = df['datetime'].dt.hour
    plot_heat_map(df)
    return df



def plot_heat_map(df) :
    df['hour'] = df['datetime'].dt.hour
    fig = plt.figure(figsize=(12, 8))
    test_1 = df.groupby(by=['month', 'hour']).count().reset_index()
    test_1 = test_1.pivot('month', 'hour', 'ip')
    # test_1 = test_1[['hour', 'month', 'ip']]
    sns.heatmap(test_1)
    plt.autoscale(enable=True)
    plt.title('This is heatmap that shows traffic trend using IP , analyzed by every hour for first of every month')

    pdf.savefig(fig)

    week_number_map = df[['day_of_week', 'month']].drop_duplicates().sort_values(by='month')
    week_number_table = render_mpl_table(week_number_map, header_columns=0, col_width=6.0)



def top_cik_file_size_anomaly(df) :

    cik_ip_filtered_df = df[['cik', 'size']]
    cik_ip_filtered_df = cik_ip_filtered_df[cik_ip_filtered_df['size'] > 0]
    top_cik = cik_ip_filtered_df.groupby('cik').count().sort_values(by='size', ascending=False).head(1).reset_index()


    top_cik_table = render_mpl_table(top_cik , header_columns=0, col_width=6.0)



    pdf.savefig(top_cik_table)

    filtered_by_top_cik = df[df['cik'] == top_cik['cik'][0]]

    filtered_by_top_cik = filtered_by_top_cik[['cik', 'size', 'quarter', 'ip']]



    filtered_by_top_cik = filtered_by_top_cik.sort_values(by=['quarter', 'size'], ascending=False).groupby(by='quarter').head(20)

    # filtered_by_top_cik

    filtered_by_top_cik = filtered_by_top_cik[filtered_by_top_cik['size'] > 0]

    fig = plt.figure(figsize=(12, 8))

    plt.title("For the top most cik we find the top 20 sized files for every quarter and plot it using a box plot. We see few anomalies when seeing few data points that lie outside the normal range")
    plt.xticks(rotation='vertical')
    plt.autoscale(enable=True, axis='both')
    sns.boxplot(x="quarter", y="size", data=filtered_by_top_cik, orient="v")
    pdf.savefig(fig)


def top_3_ip_per_month(df) :
    group_by_ip_code = df[['month', 'ip', 'code']]
    group_by_ip_code = group_by_ip_code.groupby(['month', 'ip']).count().reset_index()
    group_by_ip_code = group_by_ip_code.sort_values(by=['month', 'code'], ascending=False).groupby('month').head(3)
    objects = group_by_ip_code['ip']
    y_pos = list(range(1, 37, 1))
    performance = group_by_ip_code['code']


    fig_top_ips_per_month = plt.figure()
    barlist = plt.barh(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('Traffic Count')
    plt.title('Top 3 Ips at start of month')

    colorHandles = list()
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'silver', 'black', 'grey', 'maroon', 'navy']
    month_names = ['Dec', 'Nov', 'Oct', 'Sept', 'Aug', 'Jul', 'Jun', 'May', 'Apr', 'Mar', 'Feb', 'Jan']
    for i in range(1, 13, 1):
        barlist[i * 3 - 1].set_color(colors[i - 1])
        barlist[i * 3 - 2].set_color(colors[i - 1])
        barlist[i * 3 - 3].set_color(colors[i - 1])
        patch = mpatches.Patch(color=colors[i - 1], label=month_names[i - 1])
        colorHandles.append(patch)

    plt.legend(handles=colorHandles)

    pdf.savefig(fig_top_ips_per_month)



def weekday_vs_weekend_analysis(df) :
    day_of_week_analysis = df.groupby('day_of_week').count()['ip']
    day_of_week_analysis = day_of_week_analysis.sort_values(ascending=False)
    #plt.figure(figsize=(25, 5))
    fig = plt.figure()
    day_of_week_analysis[::-1].plot.barh()
    plt.title("This plot shoes the IP logging trend on weekdays and weekends")

    pdf.savefig(fig)

def top_cik_per_quarter(df):

    temp_new_df = df[['cik', 'quarter', 'ip']]
    grouped_temp_new_df = temp_new_df.groupby(['quarter', 'cik']).count()
    grouped_temp_new_df = grouped_temp_new_df.reset_index()
    grouped_temp_new_df = grouped_temp_new_df.sort_values(by=['quarter', 'ip'], ascending=False).groupby('quarter').head(3)

    objects = grouped_temp_new_df['cik']
    y_pos = (1, 2, 3, 7, 8, 9, 13, 14, 15, 19, 20, 21)
    performance = grouped_temp_new_df['ip']

    fig1 = plt.figure()
    barlist = plt.barh(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('Traffic Count')
    plt.title('Top 3 CIKs Per Quarter')

    barlist[0].set_color('r')
    barlist[1].set_color('r')
    barlist[2].set_color('r')

    barlist[3].set_color('b')
    barlist[4].set_color('b')
    barlist[5].set_color('b')

    barlist[6].set_color('g')
    barlist[7].set_color('g')
    barlist[8].set_color('g')

    barlist[9].set_color('y')
    barlist[10].set_color('y')
    barlist[11].set_color('y')

    red_patch = mpatches.Patch(color='red', label='Q4')
    blue_patch = mpatches.Patch(color='blue', label='Q3')
    green_patch = mpatches.Patch(color='green', label='Q2')
    yellow_patch = mpatches.Patch(color='yellow', label='Q1')
    plt.legend(handles=[yellow_patch, green_patch, blue_patch, red_patch])

    pdf.savefig(fig1)

def get_anomaly_top_100(df) :
    group_by_ip_code_anomaly = df[['month', 'ip', 'code']]
    group_by_ip_code_anomaly = group_by_ip_code_anomaly.groupby(['month', 'ip']).count().reset_index()
    group_by_ip_code_anomaly = group_by_ip_code_anomaly.sort_values(by=['month', 'code'], ascending=False).groupby('month').head(10)

    fig = plt.figure(figsize=(12, 8))


    plt.title("For every month, we are finding the frequency of logging for top 10 ips. Certain ips are seen to be logged more frequently than the others")
    plt.xticks(rotation='vertical')
    plt.autoscale(enable=True, axis='both')
    sns.boxplot(x="month", y="code", data=group_by_ip_code_anomaly, orient="v")

    pdf.savefig(fig)



def arg_parser():
    parser = argparse.ArgumentParser('CaseStudy1Part1')
    parser.add_argument('-y', '--year', help='The Year of the company for which the filling is to be downloaded.',
                        type=str)

    parser.add_argument('-k', '--awskey', help='The key to the aws account where the files is to be uploaded.',
                        type=str)
    parser.add_argument('-s', '--awssecretkey',
                        help='The secret key to the aws account where the files is to be uploaded', type=str)
    parser.add_argument('-b', '--awsbucket', help='The bucket where the files is to be uploaded', type=str)
    args = parser.parse_args()
    return args


def upload_to_s3(public_id, private_id, bucket_name) :
    data = open(os.path.join('problem2-out-data.zip'), 'rb')
    summary_data = open(os.path.join('output-summary.zip'), 'rb')
    s3 = boto3.resource(
        's3',
        aws_access_key_id=public_id,
        aws_secret_access_key=private_id,
        config=Config(signature_version='s3v4')
    )
    # creating Bucket
    #s3.create_bucket(Bucket=bucket_name)
    # Image Uploaded
    s3.Bucket(bucket_name).put_object(Key='problem2-out-data.zip', Body=data, ACL='public-read-write')
    s3.Bucket(bucket_name).put_object(Key='output-summary.zip', Body=summary_data, ACL='public-read-write')
    print('Uploaded to S3!')


# Execution starts here

if __name__ == '__main__':
    print('Starting Now!!!')
    args = arg_parser()
    year = args.year
    results = enableMultiProcessing(year)
    process_files(results)

    ACCESS_KEY_ID = args.awskey
    ACCESS_SECRET_KEY = args.awssecretkey
    BUCKET_NAME = args.awsbucket


    pdf.close()
    zipf = zipfile.ZipFile(os.path.join('output-summary.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(os.path.join('output-summary'), zipf)
    zipf.close()
    upload_to_s3(ACCESS_KEY_ID, ACCESS_SECRET_KEY, BUCKET_NAME)





























