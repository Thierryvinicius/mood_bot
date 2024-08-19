import pandas as pd
import os

def csv_to_txt(csv_file, text_column, note_column, output_txt):
    if not os.path.exists(output_txt):
        os.makedirs(output_txt)

    output_txt = os.path.join(output_txt, 'novo.txt')
    df = pd.read_csv(csv_file, sep='\t')
    
    if text_column not in df.columns or note_column not in df.columns:
        raise ValueError(f"As colunas {text_column} e/ou {note_column} não existem no arquivo CSV.")
    
    df_renamed = df.rename(columns={text_column: 'Phrase', note_column: 'Note'})
    
    df_renamed = df_renamed[['Phrase', 'Note']]
    df_renamed.to_csv('dataset/'+nome_dataset, sep='\t', index=False, header=True)
    print("Arquivo TXT gerado com sucesso!!")

csv_file = 'dataset/Restaurant_Reviews.csv'  # Caminho para o arquivo CSV contendo o dataset
text_column = 'Phrase'  # Nome da coluna no CSV que contém o texto das frases
note_column = 'Note'  # Nome da coluna no CSV que contém as notas (0 para frases negativas e 1 para frases positivas)
output_txt = '../dataset/'  # Diretório onde o arquivo TXT convertido será salvo
nome_dataset = 'restaurant_reviews.txt'  # Nome do arquivo TXT resultante

csv_to_txt(csv_file, text_column, note_column, output_txt)