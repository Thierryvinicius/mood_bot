# Projeto de Analise de sentimentos usando IA
Autores: Thierry Vinicius L. de Lima(ra188130@ucdb.br), Leonardo K. Ramos(ra184248@ucdb.br) e Pedro Eduardo F. Sampaio (ra186488@ucdb.br)

Agradecemos ao Professor Hemerson Pistori (pistori@ucdb.br) pelo código base do bot do Telegram, que foi fundamental para o desenvolvimento deste projeto.

Este projeto foi desenvolvido como parte da disciplina de Inteligência Artificial e demonstra a aplicação de redes neurais para a classificação de texto. A solução inclui um chatbot integrado ao Telegram que analisa as mensagens enviadas pelos usuários e fornece uma resposta baseada na análise de sentimento, classificando as mensagens como positivas ou negativas.

## Requisitos
Este projeto foi desenvolvido usando Python 3.10 e requer várias bibliotecas para funcionar corretamente. As instruções abaixo mostram como configurar um ambiente Conda e instalar as dependências necessárias.

### Configuração do Ambiente

1. **Crie um ambiente Conda e ative-o:**

    ```bash
    conda create -y --name chatbot python=3.10
    conda activate chatbot
    ```

2. **Instale as bibliotecas necessárias:**

    ```bash
    pip install python-telegram-bot==13.13 pillow scikit-learn nltk pandas torch
    ```

3. **Configuração do dataset:**
    vou escrever...


4. **Execução do Treinamento:**

    - Após configurar os hiperparâmetros, execute o script principal `main.py`. Este script irá iniciar o processo de treinamento do modelo, utilizando o conjunto de dados que você preparou.
    
    ```bash
    python main.py
    ```

    - O script `main.py` irá:
      - Carregar os dados pré-processados.
      - Dividir os dados em conjuntos de treinamento, validação e teste.
      - Treinar o modelo com base nos hiperparâmetros definidos.
      - Avaliar o desempenho do modelo no conjunto de validação e teste.

3. **Verificação dos Resultados:**

    - Durante a execução, o treinamento exibirá a perda (loss) para cada época e a acurácia de validação.
    - Após o treinamento, o modelo treinado será salvo no diretório  models dentro da pasta src.

Certifique-se de que todos os arquivos necessários estão corretamente configurados e os caminhos estão corretos antes de executar o script.

5. **Configuração do Chatbot do Telegram:**

    - No arquivo `telegram_bot.py`, configure o bot com seu token do Telegram.
    - Você pode configurar o token de duas maneiras:
    
    1. ```no terminal:
        python telegram_bot.py YOUR_TELEGRAM_BOT_TOKEN
    ```
    
    2. ```telegram_bot.py:
        if len(sys.argv) > 1:
            MEU_TOKEN=sys.argv[1]
        else:
            MEU_TOKEN = 'ADICIONE_SEU_TOKEN_AQUI'
    ```

## ainda vou arrumar o readme :)