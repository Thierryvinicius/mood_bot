# Exemplos botTelegram

Descrição: Exemplos de códigos em python que funcionam como chatbots para o telegram.

Autor: Hemerson Pistori (pistori@ucdb.br)

### Informações sobre como foi testado

O software foi testado usando as seguintes versões de
Sistema Operacional, Linguagem e Biblioteca do Telegram

- Sistema Operacional: Ubuntu 20.04
- Versão do python: 3.10.4
- Versão do telegram: 13.13

### Gerando o chatbot no Telegram

- Instale o app do Telegram no seu celular e crie uma conta para você
- Entre no app do telegram e procure pelo usuário @botfather
- Mande a mensagem /newbot para o @botfather e vá
respondendo às perguntas que ele fizer. 
- Anote o TOKEN que ele vai gerar para você e o use no local indicado a seguir 

### Instalando as dependências no Linux (bibliotecas)

- Abra o terminal do Linux usando CTRL+ALT+T
- Verifique os ambientes conda já instalados usando o comando
abaixo. Se já tiver o ambiente chamado chatbot, não precisa executar os comandos de instalação)
```
conda env list
```

- Se já existir o ambiente conda, use o comando abaixo
para ativá-lo
```
conda activate chatbot
```

- Se o ambiente conda chatbot não existir, execute os comandos de instalação abaixo para criar o ambiente e instalar as dependências

```
conda create -y --name chatbot python=3.10
conda activate chatbot
pip install python-telegram-bot==13.13 pillow
```

- Se não tiver conda na máquina, instale usando as instruções disponíveis aqui: https://docs.conda.io/projects/miniconda/en/latest/ (Use o Quick Command Line install para Linux)

- Para rodar o modelo criado através do TeachableMachine, será necessária instalar também as seguintes dependências:

```
pip install tensorflow-cpu

```


### Iniciando o seu chatbot

- Use o botão ao lado de 'Clone' ou o botão 'Code' (role para a parte de cima desta tela) para baixar para a sua máquina o arquivo compactado (.zip) contendo estes códigos em python
- Entre na pasta Downloads usando o comando abaixo
```
cd Downloads
```
- Use o comando abaixo para ver todos os arquivos que terminam com zip que estão na pasta Downloads para conferir
se baixou direito
```
ls *zip
```
- Descompacte o arquivo usando o comando abaixo
```
unzip 'bot_tel*'
```
- Entre na pasta descompactada
```
cd bot_telegram-master
```

- Rode um dos comandos abaixo para iniciar o seu BOT. Não esqueça de trocar pelo TOKEN que você criou anteriormente usando o @botfather

```
python coletaImagens.py SEU_TOKEN_AQUI 
python botTelegram.py SEU_TOKEN_AQUI 
```

### Próximos passos

Depois de coletar suas imagens para treinar uma inteligência artificial você pode estudar e utilizar os códigos disponíveis no link abaixo para treinar a sua inteligência artificial (use o exemplo_v4):


- [Treinando uma Inteligência Artificial](http://git.inovisao.ucdb.br/inovisao/exemplos_pytorch)


E depois de treinar a IA você pode embuti-la em uma placa Raspberry usando as orientações e os códigos disponíveis neste link: 

- [Embutindo a IA em um novo equipamento](http://git.inovisao.ucdb.br/inovisao/raspberry_camera)
