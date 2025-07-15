from statsmodels.tsa.seasonal import STL
from pyspark.sql import SparkSession
from pyspark.sql.functions import col 
from tqdm import tqdm
from scipy.stats import boxcox

import os
import warnings
import pandas as pd
import numpy as np 

warnings.filterwarnings("ignore")

class Configuration:
    def __init__(self):
        """
            Predefine must-have configuration
        """        
        
        # Initial PySpark Session
        self.spark = SparkSession \
                    .builder \
                    .appName("Anomaly Detection") \
                    .config("spark.yarn.appMasterEnv.JAVA_HOME", "/usr/lib/jvm/jre-11") \
                    .config("spark.sql.legacy.json.allowEmptyString.enabled", value=True) \
                    .enableHiveSupport() \
                    .getOrCreate()
        
        # Define paths to your custom python modules
        self.customPythonModule = [
            '/unified/user/cj/python_module/anomaly_detection/lib/util.py',
            '/unified/user/cj/python_module/anomaly_detection/lib/dataset.py',
            '/unified/user/cj/python_module/anomaly_detection/lib/model.py'
        ]
          
        # Define paths to your text files
        self.sql_files = [
            '/unified/user/cj/python_module/anomaly_detection/query/succ_rate.txt'
        ] 
           
        # window size for one hours 
        self.window_size = 289
        
        self.two_sigma = 1.96
        
        self.must_have_features = ['dt', 'tx_hour', 'count', 'mcc_mnc', 'rat_type', 'par_year', 'par_month', 'par_date', 'par_bound_type']
        
        # self.target_countries = [505,528,456,460,454,404,510,440,530,515,420,525,450,466,520,286,424,234,310,452]
        self.target_countries = [525] 
    
    def import_python(self):
        
        for python_file in self.customPythonModule:
            self.spark.sparkContext.addPyFile(python_file)

              
def read_sql_from_file(file_path:str, mcc:int) -> str: 
    """
    Read or import SQL Query from a text file.

    Args:
        file_path (STRING): the path to the Query file
        mcc (INT): Country code insert into the query

    Returns:
        str: The SQL query
    """
    
    if not isinstance(file_path, str):
        raise TypeError(f'argument: file_path supposed to be string format, not {type(file_path)}')
    
    if not isinstance(mcc, int):
        raise TypeError(f'Argument: mcc code should be integer format, not {type(mcc)}')
    
    try:
        with open(file_path, 'r') as file:
            sql_query = file.read()
            
            try:
                return sql_query.format(mcc, mcc)
            except Exception as e:
                print('Error raised when putting mcc to \{\} in query text file')
                print(f'Error: {e}')
                return None
            
    except PermissionError:
        print(f'Permission for {file_path} denied')
        return None
    except IsADirectoryError:
        print(f'{file_path} is a directory not a file')
        return None
    except Exception as e:
        print(f'Unexpected Error: {e}')    
        return None
 

def insert_into_hdfs(spark_session: SparkSession, data:pd.DataFrame):
    """
    Insert data into HDFS in parquet format.

    Args:
        spark_session (SparkSession): Spark session to use for writing data
        data (Dataframe): the data to be inserted into HDFS
    """
    
    if not isinstance(spark_session, SparkSession):
        raise TypeError(f'Argument: spark_session should be SparkSession yype, not {type(spark_session)}')
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f'Argument: data should be pd.DataFrame, not {type(data)}')
    
    try:
        spark_df = spark_session.createDataFrame(data) 
    except ValueError:
        print('Check for schema mismatch or invalid data type')
        return None
    except Exception as e:
        print(f'Unexpected Error: {e}')
        return None
 
    spark_df = spark_df.withColumn("rat_type", col("rat_type").cast("int"))
    spark_df = spark_df.withColumn("dt", col("dt").cast("timestamp")) 
    spark_df = spark_df.withColumn("success_rate", col("success_rate").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("total_count", col("total_count").cast("int")) 
    spark_df = spark_df.withColumn("feature_1", col("feature_1").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_2", col("feature_2").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_3", col("feature_3").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_4", col("feature_4").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_5", col("feature_5").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_6", col("feature_6").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_7", col("feature_7").cast("decimal(20, 6)"))
    spark_df = spark_df.withColumn("feature_8", col("feature_8").cast("decimal(20, 6)"))
    
    try:    
        spark_df.write.mode('append').partitionBy(
            'par_model', 
            'par_year', 
            'par_month', 
            'par_date', 
            'par_bound_type'
        ).parquet(
            "hdfs://hadoopha/roam352_report/data/report/data_anomaly_2"
        )
    except Exception as e: 
        print(f"Error: {e}")
        return None


def execute_detection(spark_session: SparkSession, data: pd.DataFrame, configuration): 
    """
    The main function to execute the anomaly detection process

    Args:
        spark_session (object): Spark session to use for processinng data
        data (dataframe): the data to be processed 
        configuration (object): user defined configuration or settings
    """
    
    if not isinstance(spark_session, SparkSession):
        raise TypeError(f'Argument: spark_session should be SparkSession type, not {type(spark_session)}')
     
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f'Argument: data should be pd.DataFrame, not {type(data)}')
    
    # convert to datetime
    data['dt'] = pd.to_datetime(data['dt'])
    
    # find failed transaction count
    data['failed_count'] = data['total_count'] - data['count']
      
    feature_to_target = ['count', 'failed_count']   
    
    for column in feature_to_target:
        transformed_data, lambda_param = boxcox(data[column] + 1)
        
        # decomposition of original data
        stl = STL(pd.Series(transformed_data).copy(), period=configuration.window_size, robust=True).fit() 
    
        # detect anomalies in trend component
        indicator_1, smoothed_trend = np.where(model.threshold_based_detector(
            pd.Series(stl.trend).copy(),
            configuration.window_size,
            configuration.two_sigma 
        ), 1, 0)
        
        # true trend
        trend_anomaly_idx = np.where(indicator_1 == 1)[0]
        trend_euclidean = stl.trend - smoothed_trend
        data[column + '_trend'] = np.where(indicator_1 == 1, stl.trend - trend_euclidean, stl.trend)
          
        # detect anomalies in seasonal component
        indicator_2, smoothed_seasonal = np.where(model.threshold_based_detector(
            pd.Series(stl.seasonal).copy(),
            configuration.window_size,
            configuration.two_sigma
        ), 1, 0)
        
        # true seasonal
        seasonal_anomaly_idx = np.where(indicator_2 == 1)[0]
        seasonal_euclidean = stl.seasonal - smoothed_seasonal
        data[column + '_seasonal'] = np.where(indicator_2 == 1, stl.seasonal - seasonal_euclidean, stl.seasonal)
           
           
        # gather all anomalies to residual component
        data[column + '_residual'] = stl.resid
        data[column + '_residual'][trend_anomaly_idx] = trend_euclidean[trend_anomaly_idx]
        data[column + '_residual'][seasonal_anomaly_idx] = seasonal_euclidean[seasonal_anomaly_idx]
              
        trend_strength = max(
            0, 
            1 - (np.var(data[column + '_residual']) / np.var(data[column + '_residual'] + data[column + '_trend']))
        )
        
        # seasonal_strength = max(
        #     0, 
        #    1 - (np.var(data[column + '_residual']) / np.var(data[column + '_residual'] + data[column + '_seasonal']))
        # )
        
        # residual_strength = 1 - trend_strength - seasonal_strength
         
        indicator_3, mad_z_score = model.mad_based_z_score(
            data[column + '_residual'].copy(),
            trend_strength,
            configuration.window_size
        )
        
        indicator_1 = np.where(indicator_3, 2, 0)
        data[column + '_rank_quantile'] = util.get_quantiles_series(
            mad_z_score, mad_z_score
        )
        
        # Combine result
        indicator_1_2 = np.where((indicator_1 + indicator_2) > 0, 1, 0)  
        all_outlier = np.where((indicator_1_2 + indicator_3) > 0, 1, 0)
          
        # remove anomalies not in dense
        data[column + "_expected_data"] = data[column + '_trend'] + data[column + '_seasonal']
        point_or_contextual, data[column + '_zscore'] = model.get_contextual(
            data[column].copy(),
            data[column + "_expected_data"].copy(),
            all_outlier.copy(), 
            configuration.window_size
        )
         
        data[column + '_final_result'] = all_outlier
     
    
    data['par_model'] = 'ENSEMBLE_MODEL_BOXCOX'
    
    data['is_outlier'] = data['count_final_result'].astype(str)
    
    data['feature_1'] = data['count']
    
    data['feature_2'] = data['failed_count']
     
    data['feature_3'] = data['failed_count_final_result']
    
    data['feature_4'] = data['count_residual']
    
    # data['feature_5'] = data['count_smoothed_data']
    # data['feature_5'] = STL(data['failed_count'].copy(), period=configuration.window_size // 24, robust=True).fit().trend
    data['feature_5'] = data['count_rank_quantile']
    
    # data['feature_6'] = data['failed_count_smoothed_data']
    # data['feature_6'] = STL(data['count'].copy(), period=configuration.window_size // 24, robust=True).fit().trend
    data['feature_6'] = data['failed_count_rank_quantile']
    
    data['feature_7'] = data['count_expected_data']
    # data['feature_7'] = STL(data['count_seasonal'].copy() + data['count_trend'].copy(), period=configuration.window_size // 24, robust=True).fit().trend
    # data['feature_5'] = data['count_expected_data']
    
    data['feature_8'] = data['failed_count_expected_data']
    # data['feature_8'] = STL(data['failed_count_seasonal'].copy() + data['failed_count_trend'].copy(), period=configuration.window_size // 24, robust=True).fit().trend
    # data['feature_6'] = data['count_expected_seasonal']
    
    
    data_to_ingest = data[
        [
            'dt', 
            'mcc_mnc', 
            'rat_type',  
            'success_rate', 
            'is_outlier',  
            'total_count',
            'feature_1',
            'feature_2',
            'feature_3', 
            'feature_4', 
            'feature_5', 
            'feature_6', 
            'feature_7', 
            'feature_8',
            'par_model', 
            'par_year', 
            'par_month', 
            'par_date', 
            'par_bound_type'
        ]
    ][configuration.window_size:]
     
    insert_into_hdfs(spark_session, data_to_ingest)
         
      
def start_action(spark_session: SparkSession, data_source, configuration):
    """
    Preprocess the data and group the into subset based on mcc, mnc, rat type and bound type.

    Args:
        spark_session (object): Spark session initiated 
        data_source (dataframe): data extracted from SQL query
        configuration (object): user fdefined configuration or settings
    """

    mcc_list = data_source.get_mcc_list()
    bound_list = data_source.get_bound_list()
    rat_list = data_source.get_rat_list() 
 
    for mcc_mnc in tqdm(mcc_list): 
        for bound_type in bound_list:
            for rat_type in rat_list:   
                data = data_source.get_data(mcc_mnc, bound_type, rat_type) 
                
                if data.empty:
                    print(f"===== Skipping - No Data Found in MCC-MNC: {mcc_mnc}, Bound Type: {bound_type}, RAT Type: {rat_type}") 
                    continue
                
                print(f"===== Now running - MCC-MNC: {mcc_mnc}, Bound Type: {bound_type}, RAT Type: {rat_type}") 
                 
                execute_detection(spark_session, data, configuration)
 
                 
if __name__ == "__main__":
    # Initialize SparkSession
    configuration = Configuration()
    configuration.import_python() 
    
    import dataset
    import model
    import util
    
    if not hasattr(configuration, 'window_size') or not hasattr(configuration, 'target_countries'):
        raise ValueError('Window size did not configured in global configuration')
    
    # import settings
    spark_session = configuration.spark
    sql_files = configuration.sql_files
      
    for file_path in sql_files:
        favorite_mcc = configuration.target_countries

        for mcc in favorite_mcc: 
            # Read the text file into a Spark DataFrame
            sql_query = read_sql_from_file(file_path, mcc) 
            
            try:
                # get all data from sql 
                data = dataset.Dataset( 
                    query=sql_query   
                ) 
                
                start_action(spark_session, data, configuration)
                            
            except Exception as e: 
                print(f"Error: {e}")
                    
    spark_session.stop()