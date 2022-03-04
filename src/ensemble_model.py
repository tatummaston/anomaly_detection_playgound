from mad import MAD_model
from arima import arima_model
    


def anomaly_ensemble(data_arima, data_mad, window):
    DF_m = data_mad.copy()
    df_a = data_arima.copy()

    # train arima model & get anomalies 
    arima = arima_model((3,0,2), .01, 75)
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