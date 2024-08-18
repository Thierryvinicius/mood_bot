"""
Exemplo de um chatbot para Telegram

Código disponibilizado por Karan Batra
Alterações feitas por Hemerson Pistori (pistori@ucdb.br), principalmente a parte que trata de imagens.

Como executar:
python botTelegram.py COPIE_AQUI_SEU_TOKEN

Funcionalidade: repete as mensagens de texto que alguém envia para o seu chatbot e devolve duas estatísticas das imagens quando o usuário manda uma imagem.
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
#MEU_TOKEN=sys.argv[1]
MEU_TOKEN='6616924477:AAHzgajvLT2Xp62gvn4OBnYIZ6R9LCsnVlg'

# Pasta para imagens enviadas pelo usuário
pasta_imgs='./Telegram_Imagens_Recebidas/' 

print('Carregando BOT usando o token ',MEU_TOKEN)


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define algumas respostas padrão para alguns comandos

# Resposta para quando o usuário digita um texto.
# Apenas responde com o mesmo texto que o usuário entrou
def echo(update, context):
    resposta='Você disse: '+update.message.text+' ?'
    update.message.reply_text(resposta)

# Resposta para quando o usuário mandar uma imagem
def processa_imagem(update, context):
    
    # Entra na pasta onde ficarão as imagens 
    os.chdir(pasta_imgs)
    
    # Pega o identificador da última imagem enviada 
    identificador = update.message.photo[-1].file_id
    
    # Pega o arquivo
    arquivo = context.bot.get_file(identificador)
    
    # Baixa o arquivo
    nome_imagem = arquivo.download()
    print('Processando arquivo ',pasta_imgs+nome_imagem)
    
    # Abre o arquivo como sendo uma imagem usando o PIL
    imagem = Image.open(nome_imagem).convert('RGB')
    
    # Pega algumas estatísticas dos valores dos pixels da imagem
    stat = ImageStat.Stat(imagem)
    
    # Devolve para o usuário uma mensagem de texto com duas
    # das estatísticas calculadas
    update.message.reply_text(f'Estatísticas da Imagem: valor médio dos pixels no canais R, G e B = {stat.mean}, desvio padrão = {stat.stddev}')



# Resposta para o comando /start
def start(update, context):
    update.message.reply_text('Olá, já comecei, é só escrever qualquer coisa ou mandar uma imagem')


# Resposta para o comando /help
def help(update, context):
    update.message.reply_text('Eu só sei repetir o que me falam por enquanto')



# Salva as mensagens de erro
def error(update, context):
    logger.warning('Operação "%s" causou o erro "%s"', update, context.error)


def main():

    # Cria o módulo que vai ficar lendo o que está sendo escrito
    # no seu chatbot e respondendo.
    # Troque TOKEN pelo token que o @botfather te passou (se
    # ainda não usou @botfather, leia novamente o README)
    updater = Updater(MEU_TOKEN, use_context=True)

    # Cria o submódulo que vai tratar cada mensagem recebida
    dp = updater.dispatcher

    # Define as funções que vão ser ativadas com /start e /help
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # Define a função que vai tratar os textos
    dp.add_handler(MessageHandler(Filters.text, echo))

    # Cria pasta para as imagens enviadas pelo usuário
    os.makedirs('./Telegram_Imagens_Recebidas',exist_ok=True)

    # Define a função que vai tratar as imagens
    dp.add_handler(MessageHandler(Filters.photo, processa_imagem))


    # Define a função que vai tratar os erros
    dp.add_error_handler(error)

    # Inicia o chatbot
    updater.start_polling()

    # Roda o bot até que você dê um CTRL+C
    updater.idle()


if __name__ == '__main__':
    print('Bot respondendo, use CRTL+C para parar')
    main()

