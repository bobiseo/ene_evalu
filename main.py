from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    df = get_input_data()
    return  df

def preprocess_data(df):
    df =  de_duplication(df)
    df = noise_remover(df)
    return df

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(f"Processing group: {name}")
    
        X, tfidfconverter = get_tfidf_embd(group_df)
        
        perform_modelling(X, group_df, Config.HIERARCHY)

