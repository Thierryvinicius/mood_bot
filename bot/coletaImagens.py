"""
Chatbot coletor de imagens

Autor: Hemerson Pistori (pistori@ucdb.br)

Funcionalidade: coleta imagens de diferentes classes (neste
exemplo, apenas gente e coisa) e guarda as imagens
de cada classe em uma pasta diferente, já pronta para
ser usada pelo exemplo_pytorch_v4.ipynb disponível
aqui: http://git.inovisao.ucdb.br/inovisao/exemplos_pytorch

Como executar:
python coletaImagens.py COPIE_AQUI_SEU_TOKEN

Escreva "Guarde imagens de gente" (ou qualquer frase contendo a palavra gente) e depois comece a mandar
imagens de gente para ele). Escreve "Pegue imagens de
outras coisas (ou qualquer frase contendo a palavra 
coisa) e comece a mandar fotos de coisas que não
são gente.

"""

from PIL import Image,ImageStat
import os
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import sys

# Lê o token como parâmetro na linha de comando
# Você pode também trocar diretamente aqui sys.argv[1] pelo
# seu token no telegram (ver README.md para saber como
# criar seu bot no telegram)
MEU_TOKEN=sys.argv[1]

# Pasta raiz de onde você executou o programa
pasta_raiz=os.getcwd()

# Indica qual classe de imagens está sendo coletada
# Se o usuário não falar nada, irá coletar na pasta gente
classe='gente'

# Pasta para imagens enviadas pelo usuário
pasta_imgs='/data/'+classe+'/' 

print('Carregando BOT usando o token ',MEU_TOKEN)


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define algumas respostas padrão para alguns comandos

# Quando o usuário manda um texto para o chat o
# código abaixo é executado.
def mandou_texto(update, context):
    # Tem que falar que vai usar a variável que foi
    # declara lá em cima senão ele cria uma nova
    global classe, pasta_imgs,pasta_raiz
    
    classe_valida = True
    # Verifica se o texto mandado pelo usuário contém
    # a palavra gente 
    if "gente" in update.message.text.lower():
        classe="gente"
    elif "coisa" in update.message.text.lower():
        classe="coisa"
    else:
        print('Mandou classe inválida')
        classe_valida = False
            
    if classe_valida:  
       # Nome da pasta onde devem ficar as imagens da classe 
       pasta_imgs='/data/'+classe+'/'
       # Cria pasta para as imagens enviadas pelo usuário
       os.makedirs(pasta_raiz+pasta_imgs,exist_ok=True)
       # Entra na pasta criada
       os.chdir(pasta_raiz+pasta_imgs)
       resposta='Pode começar a me mandar imagens de '+classe+' para treinarmos nossa IA !!!'     
    else:
       resposta='Você precisa escrever gente ou coisa. Como não escreveu, vou assumir que as imagens serão da classe '+classe+' por enquanto'     

    update.message.reply_text(resposta)
    

# Quando o usuário manda um texto para o chat o
# código abaixo é executado.
def mandou_imagem(update, context):
    # Tem que falar que vai usar a variável que foi
    # declara lá em cima senão ele cria uma nova
    global classe, pasta_imgs,pasta_raiz
       
    # Pega o identificador da última imagem enviada 
    identificador = update.message.photo[-1].file_id
    
    # Pega o arquivo
    arquivo = context.bot.get_file(identificador)
    
    # Baixa o arquivo
    nome_imagem = arquivo.download()
    print('Processando arquivo ',pasta_imgs+nome_imagem)
    
    update.message.reply_text('Guardei uma imagem de '+classe)



# Resposta para o comando /start
def start(update, context):
    update.message.reply_text('Olá, escreve gente ou coisa e depois comece a me mandar fotos de gente ou de coisa')


# Resposta para o comando /help
def help(update, context):
    update.message.reply_text('Olá, escreve gente ou coisa e depois comece a me mandar fotos de gente ou de coisa')



# Salva as mensagens de erro
def error(update, context):
    logger.warning('Operação "%s" causou o erro "%s"', update, context.error)


def main():

    # Cria o módulo que vai ficar lendo o que está sendo escrito
    # no seu chatbot e respondendo.
    # Troque TOKEN pelo token que o @botfather te passou (se
    # ainda não usou @botfather, leia novamente o README)
    updater = Updater(MEU_TOKEN, use_context=True)
    print('valor de updater = ',updater)
    if updater==None:
       print('TOKEN inválido, se usou run.sh tem que alterar o arquivo minha_chave.txt e colocar seu TOKEN lá !!!')
       exit(0)

    # Cria o submódulo que vai tratar cada mensagem recebida
    dp = updater.dispatcher
    print('valor de dp = ',dp)

    # Define as funções que vão ser ativadas com /start e /help
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # Define a função que vai tratar os textos
    dp.add_handler(MessageHandler(Filters.text, mandou_texto))

    # Cria pasta para as imagens enviadas pelo usuário
    os.makedirs(pasta_raiz+pasta_imgs,exist_ok=True)
    # Entra na pasta criada
    os.chdir(pasta_raiz+pasta_imgs)
    
    # Define a função que vai tratar as imagens
    dp.add_handler(MessageHandler(Filters.photo, mandou_imagem))


    # Define a função que vai tratar os erros
    dp.add_error_handler(error)

    # Inicia o chatbot
    updater.start_polling()

    # Roda o bot até que você dê um CTRL+C
    updater.idle()


if __name__ == '__main__':
    print('Bot respondendo, use CRTL+C para parar')
    main()

