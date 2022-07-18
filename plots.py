import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from collections import defaultdict

def pad_matrix(arr):
    max_len = len(max(arr, key=lambda x: len(x)))

    padded = np.zeros((len(arr), max_len), dtype=int)

    for i in range(len(arr)):
        padded[i:i + 1, 0:len(arr[i])] = arr[i]
        
    return padded

def make_plots(experiment_dir: str, dest_dir=None):
    
    if not dest_dir:
        dest_dir = os.path.join(experiment_dir, 'results_images')
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
    
    # load per-round files
    per_round_path = os.path.join(experiment_dir, 'per_round')
    per_round_stats = []

    for csv_name in os.listdir(per_round_path):
        csv_path = os.path.join(per_round_path, csv_name)
        confint_method, epsilon, alg = csv_name.split('_')[1:4]

        with open(csv_path) as f:
            rows = csv.reader(f)
            for csv_row in rows:
                df_row = [confint_method, epsilon, alg, csv_row[0], np.array([float(x) for x in csv_row[1:]])]
                per_round_stats.append(df_row)

    per_round_df = pd.DataFrame(per_round_stats, columns=['confint_method', 'epsilon', 'alg', 'field', 'data'])
    fields = per_round_df['field'].unique()

    for epsilon, df_epsilon in per_round_df.groupby(['epsilon']):

        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        fig.suptitle(f'ε = {epsilon}')
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_title('Shots_i vs. k_i')

        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_title('Queries_i vs. k_i')

        axs[2].set_yscale('log')
        axs[2].set_title('k_i vs. i')

        n_entries = 0
        
        for (alg, confint_method), df_alg_confint in sorted(df_epsilon.groupby(['alg', 'confint_method']), 
                                                            key=lambda x: (x[0][0], -ord(x[0][1][0]))):
        
            
            avg_stats = {}

            for field in fields:
                padded_stats = pad_matrix(df_alg_confint[df_alg_confint['field'] == field]['data'].to_numpy().tolist())
                div = np.count_nonzero(padded_stats, axis=0)
                div[div == 0] = 1 # suppress divide by zero
                avg_stats[field] = padded_stats.sum(axis=0) / div
                avg_stats[field][avg_stats[field] == 0] = 0.1 # for adding zero to log plots
                
            n_entries = max(n_entries, len(avg_stats['round_k']))
            label = f'{alg.upper()}+{confint_method.title()}'

            # plots for shots vs k
            axs[0].plot(avg_stats['round_k'], avg_stats['round_shots'], label=label)
            axs[0].scatter(avg_stats['round_k'], avg_stats['round_shots'])

            # plots for nqueries vs k
            axs[1].plot(avg_stats['round_k'], avg_stats['round_queries'], label=label)
            axs[1].scatter(avg_stats['round_k'], avg_stats['round_queries'])

            # plots for k
            axs[2].plot(avg_stats['round_k'], label=label)
            axs[2].scatter(range(len(avg_stats['round_k'])), avg_stats['round_k'])

        for i in range(3): axs[i].legend()

        axs[2].plot(range(n_entries), np.repeat(np.pi / 4 / float(epsilon), n_entries), c='r', linestyle='--')
        
        plt.savefig(os.path.join(dest_dir, f'{epsilon}_per_round.png'))
        plt.close()
            
    # overall query complexity and failure rate
    df_rows = []

    for csv_name in os.listdir(experiment_dir):
        csv_path = os.path.join(experiment_dir, csv_name)
        if os.path.splitext(csv_name)[1] != '.csv': continue
        amplitude, confint_method, alg = csv_name.split('_')[:3]

        with open(csv_path) as f:
            rows = csv.reader(f)
            for csv_row in rows:
                epsilon, failure, data = float(csv_row[0]), float(csv_row[1]), np.array([int(x) for x in csv_row[2:]]).mean()
                df_rows.append([alg, confint_method, epsilon, amplitude, data, failure])


    df = pd.DataFrame(df_rows, columns=['alg', 'confint_method', 'epsilon', 'amplitude', 'data', 'failure'])
    complexity = defaultdict(lambda: defaultdict(list))

    for (alg, confint_method, epsilon), df_i in sorted(df.groupby(['alg', 'confint_method', 'epsilon']), 
                                                       key=lambda x: (-x[0][2], x[0][0], -ord(x[0][1][0]))):
        complexity[(alg,confint_method)]['epsilon'].append(epsilon)
        complexity[(alg,confint_method)]['queries'].append(df_i['data'].mean())

    plt.figure(figsize=(15,7))
    plt.xscale('log')
    plt.yscale('log')

    epsilons = sorted(df['epsilon'].unique(), reverse=True)

    plt.xlim(epsilons[0]*2, epsilons[-1]/2)
    plt.title('Error vs. Number of Queries')

    for (alg, confint_method), data in complexity.items():
        epsilon, queries = data['epsilon'], data['queries']
        plt.scatter(epsilon, queries)
        plt.plot(epsilon, queries, label=f'{alg.upper()}+{confint_method.title()}')

    plt.legend()

    plt.savefig(os.path.join(dest_dir, 'complexity.png'))
    plt.close()

    # make another one for zoomed in, without log scaling


    # Query count vs. input amplitude

    amplitude_query = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for (alg, confint_method, amplitude, epsilon), df_i in sorted(df.groupby(['alg', 'confint_method', 'amplitude', 'epsilon']), 
                                                                  key=lambda x: (x[0][2], x[0][0], -ord(x[0][1][0]))):

        amplitude_query[(alg, confint_method)][epsilon]['amplitudes'].append(amplitude)
        amplitude_query[(alg, confint_method)][epsilon]['queries'].append(df_i['data'].mean())

    for (alg, confint_method) in amplitude_query:
        plt.figure()
        plt.yscale('log')

        for epsilon in amplitude_query[(alg, confint_method)]:
            amplitudes = amplitude_query[(alg, confint_method)][epsilon]['amplitudes']
            queries = amplitude_query[(alg, confint_method)][epsilon]['queries']
            plt.plot(amplitudes, queries, label='{:.0e}'.format(epsilon))
            plt.scatter(amplitudes, queries)
        plt.legend()
        plt.title(f'Number of queries vs. Input amplitude, {alg.upper()}+{confint_method.title()}')

        locs,labels = plt.xticks()
        plt.xticks(range(0, locs[-1]+1, 4)) # make more flexible

        plt.savefig(os.path.join(dest_dir, f'queries-vs-amplitude_{alg}_{confint_method}.png'))
        plt.close()


    # Failure rate

    for (amplitude, confint_method), df_i in sorted(df.groupby(['amplitude', 'confint_method'])):

        x = np.arange(len(epsilons))  # the label locations
        width = 0.35  # the width of the bars

        miae_failures = df_i[df_i['alg'] == 'miae']['failure']
        iae_failures = df_i[df_i['alg'] == 'iae']['failure']

        rects1 = plt.bar(x - width/2, miae_failures, width, label='MIAE')
        rects2 = plt.bar(x + width/2, iae_failures, width, label='IAE')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        plt.ylabel('Failure ratio')
        plt.title((f'Failure rates, a = {amplitude}, CI = {confint_method.title()}'))
        plt.xticks(x, epsilons)
        plt.ylim(0,0.25)

        plt.legend()

        ax = plt.twiny()
        line_xs = np.arange(len(epsilons)+3) - 2

        ax.plot(line_xs, [0.05] * len(line_xs), linestyle='--', c='r')
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            labeltop=False ) # labels along the bottom edge are off
        ax.set_xlim(0,5)

        plt.savefig(os.path.join(dest_dir, f'failures_{amplitude}_{confint_method}.png'))
        plt.close()