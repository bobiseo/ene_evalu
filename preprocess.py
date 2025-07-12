import pandas as pd 
import re
from Config import *
from langdetect import detect, LangDetectException
from googletrans import Translator
from sklearn.model_selection import train_test_split

translator = Translator()

def get_input_data()->pd.DataFrame:
    df1 = pd.read_csv("data//AppGallery.csv", skipinitialspace=True)
    df1.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df2 = pd.read_csv("data//Purchasing.csv", skipinitialspace=True)
    df2.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)
    df = pd.concat([df1, df2])
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    df["y"] = df[Config.CLASS_COL]
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
    return df

def de_duplication(data: pd.DataFrame):
    data["ic_deduplicated"] = ""

    cu_template = {
        "english":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team\,?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.?"]
        ,
        "german":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Kundenservice\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist eine Gesellschaft nach irischem Recht mit Sitz in Dublin, Irland\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE ist der Anbieter von Huawei Mobile Services für Huawei- und Honor-Gerätebesitzer in Europa, Kanada, Australien, Neuseeland und anderen Ländern\.?"]
        ,
        "french":
            ["L'équipe d'assistance à la clientèle d'Aspiegel\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est une société de droit irlandais dont le siège est à Dublin, en Irlande\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE est le fournisseur de services mobiles Huawei aux propriétaires d'appareils Huawei et Honor en Europe, au Canada, en Australie, en Nouvelle-Zélande et dans d'autres pays\.?"]
        ,
        "spanish":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Soporte Servicio al Cliente\,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) es una sociedad constituida en virtud de la legislación de Irlanda con su sede en Dublín, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE es el proveedor de servicios móviles de Huawei a los propietarios de dispositivos de Huawei y Honor en Europa, Canadá, Australia, Nueva Zelanda y otros países\.?"]
        ,
        "italian":
            ["Il tuo team ad (?:Aspiegel|\*\*\*\*\*\(PERSON\)),?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è una società costituita secondo le leggi irlandesi con sede a Dublino, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE è il fornitore di servizi mobili Huawei per i proprietari di dispositivi Huawei e Honor in Europa, Canada, Australia, Nuova Zelanda e altri paesi\.?"]
        ,
        "portguese":
            ["(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team,?",
             "Die (?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é uma empresa constituída segundo as leis da Irlanda, com sede em Dublin, Irlanda\.?",
             "(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE é o provedor de Huawei Mobile Services para Huawei e Honor proprietários de dispositivos na Europa, Canadá, Austrália, Nova Zelândia e outros países\.?"]
        ,
    }

    cu_pattern = ""
    for i in sum(list(cu_template.values()), []):
        cu_pattern = cu_pattern + f"({i})|"
    cu_pattern = cu_pattern[:-1]

    # -------- email split template

    pattern_1 = "(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)"
    pattern_2 = "(On.{30,60}wrote:)"
    pattern_3 = "(Re\s?:|RE\s?:)"
    pattern_4 = "(\*\*\*\*\*\(PERSON\) Support issue submit)"
    pattern_5 = "(\s?\*\*\*\*\*\(PHONE\))*$"

    split_pattern = f"{pattern_1}|{pattern_2}|{pattern_3}|{pattern_4}|{pattern_5}"

    # -------- start processing ticket data

    tickets = data["Ticket id"].value_counts()

    for t in tickets.index:
        #print(t)
        df = data.loc[data['Ticket id'] == t,]

        # for one ticket content data
        ic_set = set([])
        ic_deduplicated = []
        for ic in df[Config.INTERACTION_CONTENT]:

            # print(ic)

            ic_r = re.split(split_pattern, ic)
            # ic_r = sum(ic_r, [])

            ic_r = [i for i in ic_r if i is not None]

            # replace split patterns
            ic_r = [re.sub(split_pattern, "", i.strip()) for i in ic_r]

            # replace customer template
            ic_r = [re.sub(cu_pattern, "", i.strip()) for i in ic_r]

            ic_current = []
            for i in ic_r:
                if len(i) > 0:
                    # print(i)
                    if i not in ic_set:
                        ic_set.add(i)
                        i = i + "\n"
                        ic_current = ic_current + [i]

            #print(ic_current)
            ic_deduplicated = ic_deduplicated + [' '.join(ic_current)]
        data.loc[data["Ticket id"] == t, "ic_deduplicated"] = ic_deduplicated
    data.to_csv('out.csv')
    data[Config.INTERACTION_CONTENT] = data['ic_deduplicated']
    data = data.drop(columns=['ic_deduplicated'])
    return data

def coarse_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Language-agnostic trim that runs *before* translation.
    Removes headers, IDs, > quoting lines, excessive dashes, etc.
    Keeps the text short for Google-Translate.
    """
    ic_col = Config.INTERACTION_CONTENT

    header_regex = r"(from\s*:.*subject\s*:)|(on .* wrote:)|(-{5,}original message-{5,})"
    dash_regex   = r"\-{4,}"          # long '----' separators
    id_regex     = r"(ticket\s*id\s*:\s*\d+)|(xxxxx@xxxx\.com)|(\*\*\*\*\*\(PHONE\))"

    patterns = [header_regex, dash_regex, id_regex]

    for p in patterns:
        df[ic_col] = df[ic_col].str.replace(p, " ", regex=True)

    # squeeze multiple spaces
    df[ic_col] = df[ic_col].str.replace(r"\s{2,}", " ", regex=True).str.strip()
    return df

def noise_remover(df: pd.DataFrame):
    noise = "(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].str.lower().replace(noise, " ", regex=True).replace(r'\s+', ' ', regex=True).str.strip()
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].str.lower()
    noise_1 = [
        "(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
        "(january|february|march|april|may|june|july|august|september|october|november|december)",
        "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        "\d{2}(:|.)\d{2}",
        "(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
        "dear ((customer)|(user))",
        "dear",
        "(hello)|(hallo)|(hi )|(hi there)",
        "good morning",
        "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
        "thank you for contacting us",
        "thank you for your availability",
        "thank you for providing us this information",
        "thank you for contacting",
        "thank you for reaching us (back)?",
        "thank you for patience",
        "thank you for (your)? reply",
        "thank you for (your)? response",
        "thank you for (your)? cooperation",
        "thank you for providing us with more information",
        "thank you very kindly",
        "thank you( very much)?",
        "i would like to follow up on the case you raised on the date",
        "i will do my very best to assist you"
        "in order to give you the best solution",
        "could you please clarify your request with following information:"
        "in this matter",
        "we hope you(( are)|('re)) doing ((fine)|(well))",
        "i would like to follow up on the case you raised on",
        "we apologize for the inconvenience",
        "sent from my huawei (cell )?phone",
        "original message",
        "customer support team",
        "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
        "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
        "canada, australia, new zealand and other countries",
        "\d+",
        "[^0-9a-zA-Z]+",
        "(\s|^).(\s|$)"]
    for noise in noise_1:
        #print(noise)
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].replace(noise, " ", regex=True)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].replace(r'\s+', ' ', regex=True).str.strip()
    #print(df.y1.value_counts())
    good_y1 = df.y1.value_counts()[df.y1.value_counts() > 10].index
    df = df.loc[df.y1.isin(good_y1)]
    #print(df.shape)
    return df

# def translate_to_en(texts: list[str]) -> list[str]:
#     """
#     Translate each string in `texts` into English using googletrans.
#     If translation fails or text is already English, return original.
#     """
#     results: list[str] = []
#     for txt in texts:
#         if not txt:
#             results.append(txt)
#             continue
#         try:
#             trans = translator.translate(txt, dest='en')
#             results.append(trans.text)
#         except Exception:
#             # on error, just keep the original
#             results.append(txt)
#     return results


def translate_to_en(texts: list[str]) -> list[str]:
    """
    Return an English version of each string in `texts`.
    • If the text is already (detected as) English, return it untouched.
    • If detection fails or translation raises, fall back to the original text.
    """
    results: list[str] = []

    for txt in texts:
        clean = (txt or "").strip()
        if not clean or clean.isnumeric():          # nothing useful to translate
            results.append(clean)
            continue

        try:
            lang = detect(clean)
        except LangDetectException:
            lang = "unknown"

        if lang == "en":
            results.append(clean)
            continue

        try:
            translated = translator.translate(clean, dest="en").text
            results.append(translated)
        except Exception:
            results.append(clean)                   # graceful degradation

    return results

def merge_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ticket summary + interaction content into df["text"].
    Assumes each column is already cleaned and (if needed) translated to English.
    """
    df["text"] = (
        df[Config.TICKET_SUMMARY].fillna("") + " " +
        df[Config.INTERACTION_CONTENT].fillna("")
    ).str.strip()
    return df


#new
def load_and_split_data(test_size=0.3, val_size=0.15, stratify_col='y2'):
    """
    데이터 로드 → train/validation/test 분할하여 반환.
    ├── test_size: 전체 중 테스트 비율
    ├── val_size: 전체 중 검증(validation) 비율
    └── stratify_col: 분할 시 분포 유지할 레이블 컬럼명
    """
    df = get_input_data()

    # 1) train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=SEED
    )

    # 2) temp → val / test
    relative_val = val_size / (1 - test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,
        stratify=temp_df[stratify_col],
        random_state=SEED
    )

    return train_df, val_df, test_df  