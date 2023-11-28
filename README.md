# Case - Prevendo casos de churn em uma plataforma de streaming

## Introdução

Em uma plataforma de streaming, a diretoria está preocupada com o alto índice de usuários cancelando as suas assinaturas. Eles acreditam que é possível prever se um usuário tem mais chance de deixar a plataforma antes que isso aconteça, e com base nessa informação tomar ações para reduzir o churn. Para realizar tal trabalho, a empresa forneceu uma base de dados em csv contendo dados sobre as contas dos clientes.

## Objetivo do projeto

O objetivo é criar um modelo de classificação utilizando machine learning que seja capaz de prever se um usuário tem mais chance de cancelar a sua assinatura na plataforma ou não.

## Sobre os dados

Os dados fornecidos possuem informações sobre as contas dos clientes na plataforma de streaming, divididos entre contas Basic, Standard e Premium, onde cada uma oferece uma gama maior de serviços que a anterior.

- user_id = Código de identificação do cliente;
- age = Idade do cliente;
- gender = Gênero do cliente;
- time_on_plataform = Dias de assinatura ativa do cliente;
- subscription_type = Tipo de conta;
- avg_rating = Avaliação média dos conteúdos da plataforma;
- num_active_profiles = Número de perfis ativos na plataforma;
- num_streaming_services = Quantidade de serviços de streaming que o cliente possui;
- devices_connected = Quantidade de dispositivos conectados à conta;
- churned = Se o cliente cancelou a conta ou não.

## Bibliotecas utilizadas

Segue uma lista com as bibliotecas em Python utilizadas no desenvolvimento e resolução do case.

- pandas
- numpy
- matplotlib
- seaborn
- plotly
- sklearn

## Deployment

Para executar o projeto é necessário baixar o arquivo "case_classificacao_churn_streaming.ipynb" (código do projeto) e o database "streaming_data.csv". Feito isso, salvar ambos os arquivos na mesma pasta e executar o código utilizando programas de visualização e leituras de códigos em jupyter notebook, como Visual Studio Code por exemplo, ou algum outro software similar.

Obs: para executar o código corretamente as bibliotecas listadas na sessão anterior precisam estar instaladas em sua máquina.