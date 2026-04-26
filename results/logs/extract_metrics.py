import os
import json
import numpy as np
import pandas as pd
import sys
path = sys.argv[1]
log_files = [f for f in os.listdir(path) if f.endswith('.txt')]
total_data = []
for f in log_files:
    data = {}
    data['name'] = f
    data['completed'] = []
    with open(os.path.join(path, f), 'r') as file:
        lines = '\n'.join(file.readlines())
        json_data = json.loads(lines)
        for k, v in json_data.items():
            if k != 'lengths':
                data[k] = f" {np.mean(v)} +- {np.std(v)}"
            else:
                v = np.array(v)/35
                data[k] = f" {np.mean(v)} +- {np.std(v)}"
                data['completed'] = (v!=15)
                # if v == 525:
                #     data['completed'] += [0]
                # else:
                #     data['completed'] += [1]
        data['completed'] = np.mean(data['completed'])
    total_data.append(data)
pd.DataFrame(total_data).sort_values('name').to_csv('resultnew.csv', index=False)