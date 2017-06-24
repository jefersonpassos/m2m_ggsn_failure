import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def failure_class(log):
    if "Silently_Discarding" in log:
        return "ALERTA_FALSO"
    
    if "RADIUS_AUTHENFAILED" in log:
        return "FALHA_IDENTIFICADA"
    
    return "SEM_FALHAS"

with open('./data/datelo_23_06.txt') as file:
    logs = file.readlines()
    logs = map(str.strip, logs)
    logs = pd.Series(logs)
    logs = pd.DataFrame({'LOG': logs})

logs["Date"] = logs["LOG"].apply(lambda x: x[:20])
logs["LOG"] = logs["LOG"].apply(lambda x: x[20:].strip())
logs["Status"] = logs["LOG"].apply(failure_class)

logs.to_csv('./data/datelo_23_06.csv')