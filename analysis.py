#!/usr/bin/env python3.11
# coding=utf-8

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile
import io

# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename: str) -> pd.DataFrame:
    # tyto konstanty nemente, pomuzou vam pri nacitani
    headers = ["p1", "p36", "p37", "p2a", "weekday(p2a)", "p2b", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13a",
               "p13b", "p13c", "p14", "p15", "p16", "p17", "p18", "p19", "p20", "p21", "p22", "p23", "p24", "p27", "p28",
               "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52", "p53", "p55a",
               "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", "o", "p", "q", "r", "s", "t", "p5a"]

    # def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }

    df = pd.DataFrame()

    with zipfile.ZipFile(filename, 'r') as zip_ref:

        # pro každý rok
        for file in zip_ref.namelist():
            if file.endswith('.zip'):

                 # otevri zip file pro dany rok
                with zip_ref.open(file) as year_zip_data:
                    with zipfile.ZipFile(io.BytesIO(year_zip_data.read())) as year_zip:
                        for region_code, region_number in regions.items():
                            for csv_file in year_zip.namelist():

                                # pokud nazev csv souboru odpovida existujicimu regionu
                                if csv_file.startswith(region_number):
                                    with year_zip.open(csv_file) as f:
                                        data = pd.read_csv(f, sep=';', names=headers, encoding='cp1250', low_memory=False)
                                        data['region'] = region_code
                                        df = pd.concat([df, data])

    return df

# Ukol 2: zpracovani dat
def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:

    new_df = df.copy()

    # novy sloupec date
    new_df['date'] = pd.to_datetime(new_df['p2a'], format='%Y-%m-%d')

    # sloupce s malym poctem unikatnich hodnot
    columns_to_categorize = ["p36", "weekday(p2a)", "p6", "p7", "p8", "p9", "p10", "p11", "p12",
                             "p13a", "p13b", "p13b", "p13c", "p15", "p16", "p17", "p18", "p19",
                             "p20", "p21", "p22", "p23", "p24", "p27", "p28", "p34", "p35", "p39",
                             "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", "p52",
                             "p55a", "p57", "p58", "k", "p", "t", "p5a"]
    
    # sloupce s cisly s desetinnou carkou
    columns_to_float = ["p37", "a", "b", "d", "e", "f", "g", "o", "n", "r", "s"]

    # nastaveni sloupcu na category
    for col in columns_to_categorize:
        if col in new_df.columns and col != 'region':
            new_df[col] = new_df[col].astype('category')

    # nastaveni sloupcu na float
    new_df[columns_to_float] = new_df[columns_to_float].apply(pd.to_numeric, errors='coerce')

    # odstraneni duplicit
    new_df.drop_duplicates(subset=['p1'], keep='first', inplace=True)

    # vypis velikosti
    if verbose:
        orig_size_bytes = df.memory_usage(index=False, deep=True).sum()
        new_size_bytes = new_df.memory_usage(index=False, deep=True).sum()
        print("orig_size={:.1f} MB".format(orig_size_bytes / 1000000))
        print("new_size={:.1f} MB".format(new_size_bytes / 1000000))

    return new_df

# Ukol 3: počty nehod oidke stavu řidiče
def plot_state(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):

    # nahrada za string
    replacements = {
        4: 'pod vlivem alkoholu, obsah alkoholu v krvi do 0,99 ‰',
        5: 'pod vlivem alkoholu, obsah alkoholu v krvi 1 ‰ a více',
        6: 'nemoc, úraz apod.',
        7: 'invalida',
        8: 'řidič při jízdě zemřel (infarkt apod.)',
        9: 'pokus o sebevraždu, sebevražda'
    }
    df['p57'] = df['p57'].replace(replacements)

    # pomocny sloupec pro scitani
    df['count'] = 1

    df_grouped = df.groupby(['region', 'p57'])['count'].sum().reset_index()

    fig, axs = plt.subplots(3, 2, figsize=(14, 16))

    states_order = [replacements[i] for i in [7, 6, 5, 4, 9, 8]]

    fig.suptitle("Počet nehod dle stavu řidiče při nedobrém stavu")

    # v jednom pruchodu loopu jeden graf
    for i, state in enumerate(states_order):
        ax = axs[i // 2, i % 2]
        df_state = df_grouped[df_grouped['p57'] == state]
        sns.barplot(x='region', y='count', data=df_state, ax=ax, palette='hls', hue='region', zorder=10)

        # nastaveni vzhledu
        ax.set_title("Stav řidiče: " + state)
        ax.set_facecolor('#e9e9e9')
        ax.grid(axis='y', zorder=0, color="white")
        ax.set_xlabel('')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.tick_params(left = False) 
        ax.tick_params(bottom = False) 
        
        if i % 2 == 0:
            ax.set_ylabel('Počet nehod')
        else:
            ax.set_ylabel('')
        if i >= 4:
            ax.set_xticklabels(ax.get_xticklabels())
            ax.set_xlabel('Kraj')
        else:
            ax.set_xticklabels([])

    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()

# Ukol4: alkohol v jednotlivých hodinách
def plot_alcohol(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    
    # uprava pred vykreslenim
    df = df.dropna(subset=['p2b'])
    df['hour'] = df['p2b'] // 100
    df['alcohol_present'] = df['p11'].apply(lambda x: 'Ano' if x >= 3 else 'Ne' if x in [1, 2] else 'Nic')

    # volba kraje a oriznuti hodin od 0 do 23
    regions_to_plot = ['JHM', 'MSK', 'OLK', "ZLK"]
    df = df[df['region'].isin(regions_to_plot)]
    df = df[(df['hour'] >= 0) & (df['hour'] <= 23) & (df['alcohol_present'].isin(['Ano', 'Ne']))]

    df_grouped = df.groupby(['region', 'hour', 'alcohol_present']).size().reset_index(name='accidents')

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    axs = axs.flatten()

    # v jednom pruchodu loopu jeden graf
    for ax, region in zip(axs, regions_to_plot):
        df_region = df_grouped[df_grouped['region'] == region]

        sns.barplot(x='hour', y='accidents', hue='alcohol_present', data=df_region, ax=ax, zorder=10)

        # nastaveni vzhledu
        ax.set_title("Kraj: " + region)
        ax.set_ylim(0, 3000)

        ax.set_facecolor('#f2f2f2')
        ax.grid(axis='y', zorder=0, color="white")
        ax.set_xlabel('')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.tick_params(left = False) 
        ax.tick_params(bottom = False) 
        
        ax.set_ylabel('Počet nehod')
        ax.set_xlabel('Hodina')

    for ax in axs:
        ax.get_legend().remove()

    handles, labels = axs[0].get_legend_handles_labels()

    # legenda
    fig.legend(handles, labels, bbox_to_anchor=(1.1, 0.5), loc='center right',
               title="Alkohol", frameon=False)

    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()

# Ukol 5: Zavinění nehody v čase
def plot_fault(df: pd.DataFrame, fig_location: str = None,
               show_figure: bool = False):
    
    # volba kraje
    selected_regions = ['JHM', 'MSK', 'OLK', 'ZLK']
    df = df[df['region'].isin(selected_regions)]

    # vyfiltrovani typu zavineni
    df = df[df['p10'].isin([1, 2, 3, 4])]

    # prevedeni na string
    fault_dict = {1: 'Řidičem motorového vozidla',
                  2: 'Řidičem nemotorového vozidla',
                  3: 'Chodcem',
                  4: 'Zvířetem'} 
    df['p10'] = df['p10'].map(fault_dict)

    df = df.pivot_table(index='date', columns=['region', 'p10'], aggfunc='size', fill_value=0)
    df.columns = df.columns.map('{0[0]}_{0[1]}'.format) 
    df = df.resample('M').sum().stack().reset_index()

    # prejmenovani columns
    df = df.rename(columns={'level_1': 'region_p10', 0: 'accidents'})

    # znovuvytvoreni sloupcu region a p10
    df[['region', 'p10']] = df['region_p10'].str.split('_', expand=True)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axs = axs.flatten()

    # v jednom pruchodu loopu jeden graf
    for i, (ax, region) in enumerate(zip(axs, selected_regions)):
        data = df[df['region'] == region]
        sns.lineplot(x='date', y='accidents', hue='p10', data=data, ax=ax, zorder=10, palette='tab10')

        # nastaveni vzhledu
        ax.set_title("Kraj: " + region)
        ax.set_xlim([pd.Timestamp('2016-01-01'), pd.Timestamp('2023-01-01')])
        ax.set_ylim([0, 800])  # Nastavení rozsahu na všech y osách od 0 do 800
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))

        ax.set_facecolor('#f2f2f2')
        ax.grid(axis='y', zorder=0, color="white")
        ax.grid(axis='x', zorder=0, color="white")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.tick_params(left = False) 
        ax.tick_params(bottom = False) 

        # y label pouze nalevo
        if i % 2 == 0:
            ax.set_ylabel('Počet nehod')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        # x label pouze dole
        if i >= 2:
            ax.set_xlabel('Datum')
        else:
            ax.set_xlabel('')

    for ax in axs:
        if ax.get_legend():
            ax.get_legend().remove()

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels, bbox_to_anchor=(1.25, 0.5), loc='center right',
            title="Zavinění", frameon=False)

    plt.tight_layout()

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.

    df = load_data("data/data.zip")

    df2 = parse_data(df, True)

    plot_state(df2, "01_state.png", True)

    plot_alcohol(df2, "02_alcohol.png", True)

    plot_fault(df2, "03_fault.png", True)
