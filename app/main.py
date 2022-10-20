import faulthandler

if __name__ ==  "__main__":
    from training import Preprocessing
    from compute import Compute
    faulthandler.enable() #start @ the beginning
    preprocessing = Preprocessing()
    compute = Compute()
    processing = input("Pré-processing requis ? True or False")
    # Import scrapped website content
    if processing == 'True':
        scrapped, hs_code = preprocessing.read_data()
        tokenize_sentences = preprocessing.tokenize_sentence(scrapped)
        clear_sentences = preprocessing.clear_sentence(tokenize_sentences)
    else:
        clear_sentences = preprocessing.import_consolided_data()
    training = input("Entraînement des modèles requis ? True or False ")
    if training == 'True':
        compute.train_model(clear_sentences)
    # TODO:
    #  insert cpf nomenclature in final_sentences
    modele = input("Modèle pour le calcul des scores ? 1, 2 ou 3")
    model = compute.choose_model(modele)


"""
    #CALCUL SCORES
    #df_final = pd.read_pickle('df_content_cpf.pkl')
    #df_final = pd.read_csv('data_cpf_content.csv', delimiter = "!", skiprows=range(1, 6000000), nrows=1000000)
    df_final["score"] = ""
    
    #df_final = df_final.head(8000000)
    df_final = df_final.iloc[23000000:23963605]
    print(df_final)
    #df_final = df_final[['score', 'description', 'web']]
    text_data = compute.tache_parallele(df_final, compute.get_score, multiprocessing.cpu_count() - 1)
    text_data.to_csv('scores_final_3_1.csv', sep="!")
    """
