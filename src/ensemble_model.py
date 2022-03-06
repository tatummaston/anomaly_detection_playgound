from mad import MAD_model
from arima import Arima_model
from etl import aggregate_data
import pandas as pd
    

class Ensemble():
    """
    Final Model that takes in data and a window size, preprocesses the data, and outputs the final prediction based on the two model's predictions 
    """

    def anomaly_ensemble(self, data, window):
        
        data = aggregate_data(data, window)
        
        DF_m = pd.DataFrame(data)
        df_a = pd.DataFrame(data)

        # train arima model & get anomalies 
        arima = Arima_model((3,0,2), .01, 75)
        arima_anomalies = arima.arima_model_anomalies(df_a)

        # train mad model & get anomalies 
        mad = detect_anomaly()
        mad_anomalies = mad.MAD_anomalies(DF_m)

        DF['anomaly'] = list(np.zeros(shape=(1,len(DF)-len(arima_anomalies)))[0]) + arima_anomalies
        df['anomaly'] = 0
        count = window

        for i in range(len(DF)):
            if DF.loc[i,'anomaly'] == 1:
                df.loc[(i*window):(i*window)+window,'anomaly'] = 1

        all_anom = df.anomaly
        final_preds = []
        for i in range(len(df)):
            if mad_anomalies[i] == 1 and all_anom[i] ==1:
                final_preds.append(1)
            else:
                final_preds.append(0)

        return final_preds