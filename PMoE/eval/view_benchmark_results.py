"""source: https://github.com/dotchen/LearningByCheating"""
# import re
# import numpy as np
# import pandas as pd
# from terminaltables import DoubleTable
# from pathlib import Path


# def main(path_name):

#     performance = dict()

#     path = Path(path_name)
#     for summary_path in path.glob("*/summary.csv"):
#         name = summary_path.parent.name
#         match = re.search(
#             "^(?P<suite_name>.*Town.*-v[0-9]+.*)_seed(?P<seed>[0-9]+)", name
#         )
#         suite_name = match.group("suite_name")
#         seed = match.group("seed")

#         summary = pd.read_csv(summary_path)

#         if suite_name not in performance:
#             performance[suite_name] = dict()

#         performance[suite_name][seed] = (summary["success"].sum(), len(summary))

#     table_data = []
#     for suite_name, seeds in performance.items():

#         successes, totals = np.array(list(zip(*seeds.values())))
#         rates = successes / totals * 100

#         if len(seeds) > 1:
#             table_data.append(
#                 [
#                     suite_name,
#                     "%.1f ± %.1f" % (np.mean(rates), np.std(rates, ddof=1)),
#                     "%d/%d" % (sum(successes), sum(totals)),
#                     ",".join(sorted(seeds.keys())),
#                 ]
#             )
#         else:
#             table_data.append(
#                 [
#                     suite_name,
#                     "%d" % np.mean(rates),
#                     "%d/%d" % (sum(successes), sum(totals)),
#                     ",".join(sorted(seeds.keys())),
#                 ]
#             )

#     table_data = sorted(table_data, key=lambda row: row[0])
#     table_data = [("Suite Name", "Success Rate", "Total", "Seeds")] + table_data
#     table = DoubleTable(table_data, "Performance of %s" % path.name)
#     print(table.table)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()

#     parser.add_argument("path", help="path of benchmark folder")

#     args = parser.parse_args()
#     main(args.path)
import csv
import numpy as np
from tabulate import tabulate
from itertools import product
from collections import defaultdict

TOWNS = ['Town01', 'Town02']
TRAFFICS = ['empty', 'regular', 'dense']
WEATHERS = {
    1: 'train', 3: 'train', 6: 'train', 8: 'train',
    10: 'test', 14: 'test',
}

def parse_results(path):
    
    finished = defaultdict(lambda: [])
    
    with open(path+'.csv', 'r') as file:
        log = csv.DictReader(file)
        for row in log:
            finished[(
                row['town'],
                int(row['traffic']),
                WEATHERS[int(row['weather'])],
            )].append((
                float(row['route_completion']),
                int(row['lights_ran']),
                float(row['duration'])
            ))
    
    
    for town, weather_set in product(TOWNS, set(WEATHERS.values())):
        
        output = "\n"
        output += "\033[1m========= Results of {}, weather {} \033[1m=========\033[0m\n".format(
                town, weather_set)
        output += "\n"
        list_statistics = [['Traffic', *TRAFFICS], [args.metric]+['N/A']*len(TRAFFICS),['Duration']+['N/A']*len(TRAFFICS)]
    
        
        for traffic_idx, traffic in enumerate(TRAFFICS):
            runs = finished[town,TRAFFICS.index(traffic),weather_set]

            if len(runs) > 0:
                route_completion, lights_ran, duration = zip(*runs)

                mean_lights_ran = np.array(lights_ran)/np.array(duration)*3600

                if args.metric == 'Success Rate':
                    list_statistics[1][traffic_idx+1] = "{}%".format(100*round(np.mean(np.array(route_completion)==100), 2))
                elif args.metric == 'Route Completion':
                    list_statistics[1][traffic_idx+1] = "{}%".format(round(np.mean(route_completion), 2))
                elif args.metric == 'Lights Ran':
                    list_statistics[1][traffic_idx+1] = "{} per hour".format(round(np.mean(mean_lights_ran), 2))
                
                list_statistics[2][traffic_idx+1] = "{}s".format(round(np.mean(duration), 2))
                
        output += tabulate(list_statistics, tablefmt='fancy_grid')
        output += "\n"

        print(output)
            
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--metric', default='Success Rate', choices=[
        'Success Rate', 'Route Completion', 'Lights Ran'
    ])
    
    args = parser.parse_args()
    
    parse_results(args.config)

