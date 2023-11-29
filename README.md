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

## Análise dos dados

### Idade

- Histograma das idades dos clientes

![age_histplot.png](graph_results/age_histplot.png)

É possível observar um comportamento quase que simétrico dos dados, com uma distribuição homogênea ao longo de todas as idades, sem muito destaque para nenhuma faixa etária. Estamos falando da fase adulta, abrangendo desde pessoas no período universitário, de graduação e início da carreira (20-30 anos) até pessoas que já estão no auge de sua carreia/momento financeiro mais estável (40-60 anos). Isto também demonstra que não existe um público alvo muito marcante na empresa.

### Tempo de contrato ativo

- Histograma do tempo de contrato por cliente

![time_on_platform_histplot.png](graph_results/time_on_platform_histplot.png)

Observa-se a uma massa bem distribuída de clientes que estão em contrato entre 2000 até mais de 8000 dias.

Contudo, importante destacar que esta é coluna apresenta muitos dados nulos.Logo, esta análise deve ser feita com cautela, considerando que estes dados ao serem tratados, podem passar uma visão diferente desta realidade.

### Aparelhos conectados

- Histograma dos aparelhos conectados ao serviço de streaming por cliente

![devices_conected_histplot.png](graph_results/devices_conected_histplot.png)

O resultado do gráfico acima mostra um equilíbrio muito grande no número de aparelhos conectados por cliente, mostrando uma pluralidade de perfis e faixas sociais muito grande entre os clientes da empresa. Isto corrobora fortemente com a análise feita com as faixas etárias anteriormente, onde ficou claro a existência de clientes com 20 até mais de 60 anos.

### Quantidade de serviços de streaming contratados

- Histograma da quantidade de serviços de streaming contratados por cliente

![num_streaming_services_histplot.png](graph_results/num_streaming_services_histplot.png)

Observa-se uma distribuição muito parelha também na quantidade de serviços de streaming por cliente, com dados praticamente iguais. Também casa com a pluradidade de possíveis configurações familiares para a vasta faixa etária de clientes desta empresa.

Pelo gráfico acima, conforme previsto existem alguns outliers nesta coluna. Como não são muitos, não são provenientes de erros e também não são dados completamente da fora da realidade possível desta coluna (número de serviços de streaming assinados pelo cliente), optou-se por não excluí-los nem fazer qualquer tipo de tratamento ou transformação com eles.

### Número de perfis ativos na plataforma de streaming

- Histograma da distribuição do número de perfis ativos na plataforma de streaming

![num_active_profiles_histplot.png](graph_results/num_active_profiles_histplot.png)

Aqui existe um equilíbrio muito grande, com dados praticamente iguais na quantidade de perfis. O que também reforça a tese da distribuição das idades, onde ficou claro que não existe um nicho específico de cliente, abordando diversos tipos de cliente e configurações familiares. Existem tanto clientes que moram com outras pessoas, quanto muitos que são solteiros ou moram sozinhos, e por consequência tem apenas 1 perfil em suas contas.

### Avalição média

- Histograma da distribuição de avaliações dos clientes ao serviço de streaming

![avg_rating_histplot.png](graph_results/avg_rating_histplot.png)

Da mesma forma que as demais colunas, aqui também observa-se um equilíbrio muito grande nas avaliações dadas pelos clientes, com uma média das notas igual a 3. Do ponto de vista do negócio, este não é um resultado a ser bem visto pela empresa pois demonstra que as opiniões estão muito dividas com relação aos conteúdos da plataforma, onde o ideal seria uma maior concentração de notas nas faixas mais altas, de 4 a 5.

### Churned

- Gráfico de pizza da distribuição dos casos de churn e não churn no dataset

![churned_pie_chart.png](graph_results/churned_pie_chart.png)

Pelo resultado acima percebe-se que existe uma frequência maior de clientes que não viraram churn (0.0) do que clientes que são churn (1.0). Contudo, é justificável a preocupação do gestor da empresa, pois o percentual de churn está de fato bem aumento, com 26.1% dos dados não nulos.