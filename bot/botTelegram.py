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
import torch
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.model import Net
from src.config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from src.inference import predict_sentiment


# Lê o token como parâmetro na linha de comando
# Você pode também trocar diretamente aqui sys.argv[1] pelo
# seu token no telegram (ver README.md para saber como
# criar seu bot no telegram)
if len(sys.argv) > 1:
    MEU_TOKEN=sys.argv[1]
else:
    MEU_TOKEN = 'SEU_TOKEN'

print('Carregando BOT usando o token ',MEU_TOKEN)

#Path
MODEL_PATH = os.path.abspath('src/models/text_classifier.pth')
VECTOR_PATH = os.path.abspath('src/models/tfidf_vectorizer.pkl')


#Load the model
model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

#Load the vector
with open(VECTOR_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define algumas respostas padrão para alguns comandos

# Resposta para quando o usuário digita um texto.
# Apenas responde com o mesmo texto que o usuário entrou
def echo(update, context):
    user_message = update.message.text
    sentiment = predict_sentiment(model,vectorizer, user_message)
    answer = f"A análise de sentimento para a mensagem é '{user_message}' é: '{sentiment}'"
    update.message.reply_text(answer)

# Resposta para o comando /start
def start(update, context):
    update.message.reply_text('Olá, sou o Mood Bot, um bot de análise de sentimento!')


# Resposta para o comando /help
def help(update, context):
    update.message.reply_text('Olá, envie uma mensagem e eu irei analisar ela!')


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


    # Define a função que vai tratar os erros
    dp.add_error_handler(error)

    # Inicia o chatbot
    updater.start_polling()

    # Roda o bot até que você dê um CTRL+C
    updater.idle()


if __name__ == '__main__':
    print('Bot respondendo, use CRTL+C para parar')
    main()

