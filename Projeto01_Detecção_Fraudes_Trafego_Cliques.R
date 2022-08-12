# Capitulo 22 - Projeto 1 - Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile

# Data Description = For this competition, your objective is to predict whether a user will download an app after clicking a mobile app advertisement.

# File descriptions:
# train.csv - the training set
# train_sample.csv - 100,000 randomly-selected rows of training data, to inspect data before downloading full set
# test.csv - the test set
# sampleSubmission.csv - a sample submission file in the correct format

# UPDATE: test_supplement.csv - This is a larger test set that was unintentionally released at the start of the competition. It is not necessary to use this data, but it is permitted to do so. The official test data is a subset of this data.

# Data fields = Each row of the training data contains a click record, with the following features.

# ip: ip address of click. 
# app: app id for marketing.
# device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# os: os version id of user mobile phone
# channel: channel id of mobile ad publisher
# click_time: timestamp of click (UTC)
# attributed_time: if user download the app for after clicking an ad, this is the time of the app download
# is_attributed: the target that is to be predicted, indicating the app was downloaded

# Obs: Note that ip, app, device, os, and channel are encoded.

# The test data is similar, with the following differences:

# ick_id: reference for making predictions
# is_attributed: not included

# Definindo o diretório de trabalho
setwd("D:/Cursos/Curso_FCD/1-BigDataRAzure/Arquivos/Cap22/Projeto01")
getwd()

# Carregando Pacotes
library(data.table)
library(C50)
library(caret)
library(ROCR)
library(pROC)
library(ROSE)
library(dplyr)
library(e1071)

search()
## Etapa 1 - Coletando os Dados
# Coletando Dados de amostra que sera usao em todas as etapas de pre-processamento para homologar a programação.  

###------>>>> Optei por usar apenas os dados de amostra devido a limitação de processamento de SO da minha máquina.
df_train_sample <- fread("train_sample.csv", stringsAsFactors = TRUE, sep = ",", header =T)
View(df_train_sample)
dim(df_train_sample)
str(df_train_sample)

## Etapa 2 - Pré-Processamento
# A- Remover as colunas attributed_time, click_time
# B- Checar NAs nas demais colunas

# 2-A) Removendo colunas de data

###------>>>>  Remoção das colunas de data não haver impacto para o modelo
df_train_sample$attributed_time <- NULL
df_train_sample$click_time <- NULL
dim(df_train_sample)
#Transforma variávels target em fator
df_train_sample$is_attributed <- as.factor(df_train_sample$is_attributed)
str(df_train_sample)


# 2-B)Checando valores missing
sapply(df_train_sample, function(x)sum(is.na(x)))
# Não foram encontrados NAs 

# 2-C) Soma a quantidade de cliques por IP para formar ranking. Espera-se que o dispositivo(IP) clique poucas vezes para fazer o download, 
# caso contrario é uma fraude
df_qtdeIP <- as.data.frame(df_train_sample %>%
              group_by(ip) %>%
                summarise(qtdeIP = n()))
#2-D) Cria níveis de Qtde cliques por IP 

# Função para criar níveis de cliques por IP
fac_qtdeIP <- function(qtd){
  
  if (qtd == 1) {
      return(as.factor("1"))
    }else if (qtd == 2 | qtd == 3 ) {
      return(as.factor("2"))
    }else { 
      return(as.factor("3"))
    }
    
  }

#Aplica a função fac_qtdeIP no data set df_qtdeIP criando a coluna nivQtIp
df_qtdeIP$nivQtIp <- sapply(df_qtdeIP$qtdeIP,fac_qtdeIP)

View(df_qtdeIP)
str(df_qtdeIP)

#2-E) Une o Data Fame de df_qtdeIP(ranking e niveis) ao df_train_sample
df_train_sample <- left_join(df_train_sample, df_qtdeIP) 
View(df_train_sample)


# Etapa 3 - Feature Selection com dados de amostra

# Modelo randomForest para criar um plot de importância das variáveis
library(randomForest)

modelo_sample <- randomForest( is_attributed ~ . , 
                        data = df_train_sample, 
                        ntree = 100, nodesize = 10, importance = T)

varImpPlot(modelo_sample)

#Remove as variáveis meno relavantes, inclusive as variáveis criadas, pois não apresentaram relevancia.
df_train_sample$device <- NULL
df_train_sample$os <- NULL
df_train_sample$qtdeIP <- NULL
df_train_sample$nivQtIp <- NULL

View(df_train_sample)
str(df_train_sample)


# Etapa 4 Divisão em Treino e Teste

###------>>>> Realizado a divisão dos dados entre treino e teste neste momento porque a variável Target esta desbalanceada.

# Índice de divisão dos dados
indice_Divisao <- sample(x = nrow(df_train_sample),
                              size = 0.8 * nrow(df_train_sample),
                              replace = FALSE)
View(indice_Divisao)

#APLICAND0 O INDICE
dados_treino <- df_train_sample[indice_Divisao,] 
dados_teste <- df_train_sample[-indice_Divisao,] 

str(dados_treino)
dim(dados_treino)
View(dados_treino)
dim(dados_teste)
View(dados_teste)

# Etapa 5 - Balanceamento de classe com ROSE
# Balanceamento doa dados de treino
dados_treino_bal <- ROSE(is_attributed ~ . , data = dados_treino, seed = 1)$data
prop.table(table(dados_treino_bal$is_attributed))

# Balanceamento doa dados de teste
dados_teste_bal <- ROSE(is_attributed ~ ., data = dados_teste, seed = 1)$data
prop.table(table(dados_teste_bal$is_attributed))

dim(dados_treino_bal)
dim(dados_teste_bal)

#Grava os datasets de treino e teste balanceados em csv
fwrite(dados_treino_bal, "dados_treino_bal.csv")
fwrite(dados_teste_bal, "dados_teste_bal.csv")

# Etapa 5 - Criando o modelo Decision Trees com o pacote C5.0

#Lê os datasets de treino e teste balanceados
dados_treino_bal <- fread("dados_treino_bal.csv", stringsAsFactors = TRUE, sep = ",", header =T)
dados_teste_bal <- fread("dados_teste_bal.csv", stringsAsFactors = TRUE, sep = ",", header =T)
str(dados_treino_bal)

# Modelo usando dados de treino
modelo <- C5.0(is_attributed ~ ., data = dados_treino_bal)

#Previsões usando dados de teste
previsoes <- predict(modelo, dados_teste_bal)

# Acurácia
caret::confusionMatrix(dados_teste_bal$is_attributed, previsoes, positive = '1')
###------>>>> Neste Modelo foi atingida a Acuracia de 0.8396    

# Calculamos o Score AUC
roc.curve(dados_teste_bal$is_attributed, previsoes, plotit = T, col = "red")

###------>>>> Neste Modelo foi atingido  o Score AUC de 0.840 

# Etapa 6 - Criando o modelo SVM com o pacote e1071
modelo_SVM <- svm(is_attributed ~ ., data = dados_treino_bal, scale = TRUE)
summary(modelo_SVM)
print(modelo_SVM)

previsoes_SVM <- predict(modelo_SVM, newdata = dados_teste_bal)

# Matriz de confusão
caret::confusionMatrix(previsoes_SVM, dados_teste_bal$is_attributed)
###------>>>> Neste Modelo foi atingida a Acuracia de  0.8408   

# Calculamos o Score AUC
roc.curve(dados_teste_bal$is_attributed, previsoes_SVM, plotit = T, col = "red")
###------>>>> Neste Modelo foi atingido  o Score AUC de 0.841       

# Etapa 7 - Criando o modelo KNN

#Arquivo de Controle
crtl <- trainControl(method = 'repeatedcv', repeats = 3)

# Criação do modelo
knn <- train(is_attributed ~ . ,
                data = dados_treino_bal,
                method = 'knn',
                trControl = crtl,
                tuneLength = 20)

# Modelo KNN
knn

# Número de Vizinhos x Acurácia
plot(knn)

# Fazendo previsões
Previsao_KNN <- predict(knn, newdata = dados_teste_bal)
Previsao_KNN

#Criando a Confusion Matrix
confusionMatrix(Previsao_KNN, dados_teste_bal$is_attributed)
###------>>>> Neste Modelo foi atingida a Acuracia de 0.7382   

# Calculamos o Score AUC
roc.curve(dados_teste_bal$is_attributed, Previsao_KNN, plotit = T, col = "red")
###------>>>> Neste Modelo foi atingido  o Score AUC de 0.738    

# Conclusão:
# Os modelos C5.0 e SVM apresentaram performance semelhantes e ambos podem ser utilizados para previsões deste projeto.
