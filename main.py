from data.preprocessing_data import create_dataset_sent140, rename_users
import sys
import pandas as pd
from actors.conductor import *

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--preprocessed':
            print('Read from ./csv/01-preprocessed.csv')
            df = pd.read_csv('./csv/01-preprocessed.csv')
            df = rename_users(df)
        elif sys.argv[1] == '--renamed':
            print('Read from ./csv/02-preprocessed_renamed.csv')
            df = pd.read_csv('./csv/02-preprocessed_renamed.csv')
    else:
        df = create_dataset_sent140()
        df = rename_users(df)

    df['date'] = df['date'].astype('datetime64[ns]')
    df['round'] = -1

    clients = ['Monica', 'Phoebe', 'Rachel', 'Chandler', 'Joey', 'Ross']
    conductor = Conductor(df.loc[:, ['date', 'preprocessed_tweet', 'renamed_user', 'round']], clients)
    conductor.run_rounds()




