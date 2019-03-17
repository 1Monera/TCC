### Libraries
library(readr)
library(magrittr)
library(knitr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(ggridges)
library(purrr)
library(keras)
library(kerasR)
library(Metrics)
library(glmnet)
### Leitura do Banco de Dados
setwd("~/Documents/Faculdade/Semestres/2018.1/Trabalho de Graduação 1/Execução")
banco=read_csv("creditcard.csv",col_types=list(Time=col_number()))
aa=as.data.frame(banco)
### Separação em Treino, Validação, Corte e Teste
set.seed(57)
q=sample(x=c("b1","b2","b3","b4"),
         size=nrow(aa),
         replace=TRUE,
         prob=c(0.5,0.2,0.2,0.1))
df_train=aa[q=="b1",]
df_valid=aa[q=="b2",]
df_corte=aa[q=="b3",]
df_teste=aa[q=="b4",]

### Manipulações

## Manipulação 1
get_desc = function(x){
  map(x, ~list(
    min=min(.x),
    max=max(.x),
    mean=mean(.x),
    sd=sd(.x)
  ))
}
normalization_minmax = function(x,desc) {
  map2_dfc(x, desc, ~(.x - .y$min)/(.y$max - .y$min))
}
desc=df_train %>% select(-Class) %>% get_desc()
#
x_train_1=df_train %>% select(-Class) %>% normalization_minmax(desc) %>% as.matrix()
x_valid_1=df_valid %>% select(-Class) %>% normalization_minmax(desc) %>% as.matrix()
x_corte_1=df_corte %>% select(-Class) %>% normalization_minmax(desc) %>% as.matrix()
x_teste_1=df_teste %>% select(-Class) %>% normalization_minmax(desc) %>% as.matrix()
y_train_1=df_train$Class
y_valid_1=df_valid$Class
y_corte_1=df_corte$Class
y_teste_1=df_teste$Class

## Manipulação 2
treino_2=aa[q %in% c("b1","b2"),]
x_corte_2=df_corte %>% select(-Class)
x_teste_2=df_teste %>% select(-Class)
y_corte_2=df_corte %>% select(Class)
y_teste_2=df_teste %>% select(Class)

## Manipulação 3
treino_3=aa[q %in% c("b1","b2"),-c(30:31)]
treino_3$VR=aa[q %in% c("b1","b2"),30]*ifelse(aa[q %in% c("b1","b2"),31]==0,0.01,-1)
#
x_corte_3=df_corte %>% select(-Class,-Amount)
x_teste_3=df_teste %>% select(-Class,-Amount)
y_corte_3=df_corte$Class
y_teste_3=df_teste$Class

### Construção dos Modelos
### Modelos Tipo 1
## Modelo 1
mod1=keras_model_sequential()
mod1 %>%
  layer_dense(units=15,activation="tanh",input_shape=ncol(x_train_1)) %>%
  layer_dense(units=10,activation="tanh") %>%
  layer_dense(units=15,activation="tanh") %>%
  layer_dense(units=ncol(x_train_1))
mod1 %>% compile(
  loss="mean_squared_error", 
  optimizer="adam"
)
summary(mod1)
#
set.seed(57) # 6.22 min # Epoch 42
checkpoint=callback_model_checkpoint(
  filepath="mod1_AutoEncoderPrev.hdf5",
  save_best_only=TRUE,
  period=1,
  verbose=1
)
early_stopping=callback_early_stopping(patience=5)
#mod1 %>% fit(
#  x=x_train_1[y_train_1==0,],
#  y=x_train_1[y_train_1==0,],
#  epochs=100,
#  batch_size=32,
#  validation_data=list(x_valid_1[y_valid_1==0,],x_valid_1[y_valid_1==0,]),
#  callbacks=list(checkpoint,early_stopping)
#)
mod1=load_model_hdf5("mod1_AutoEncoderPrev.hdf5",compile=FALSE)
#
pred_mod1=predict(mod1,x_corte_1)
sse_mod1=apply((pred_mod1-x_corte_1)^2,1,sum)
prev_mod1=predict(mod1,x_teste_1)
sse_mod1_teste=apply((prev_mod1-x_teste_1)^2,1,sum)
## Modelo 2
mod2=keras_model_sequential()
mod2 %>%
  layer_dense(units=15,activation="tanh",input_shape=ncol(x_train_1)) %>%
  layer_dense(units=10,activation="tanh") %>%
  layer_dense(units=15,activation="tanh") %>%
  layer_dense(units=ncol(x_train_1),activation="tanh") %>%
  layer_dense(units=1,activation="sigmoid")
mod2 %>% compile(
  loss="binary_crossentropy",
  optimizer="adam"
)
summary(mod2)
#
set.seed(57)
checkpoint=callback_model_checkpoint(
  filepath="mod2_AutoEncoderPrev.hdf5",
  save_best_only=TRUE,
  period=1,
  verbose=1
)
early_stopping=callback_early_stopping(patience=5)
#mod2 %>% fit(
#  x=x_train_1,
#  y=y_train_1,
#  epochs=100,
#  batch_size=32,
#  validation_data=list(x_valid_1,y_valid_1),
#  callbacks=list(checkpoint,early_stopping)
#)
mod2=load_model_hdf5("mod2_AutoEncoderPrev.hdf5",compile=FALSE)
pred_mod2=predict(mod2,x_corte_1)
prev_mod2=predict(mod2,x_teste_1)
## Modelo 3
mod3=keras_model_sequential()
mod3 %>%
  layer_dense(units=15,activation="relu",input_shape=ncol(x_train_1)) %>%
  layer_dense(units=15,activation="relu") %>%
  layer_dense(units=15,activation="relu") %>%
  layer_dense(units=1,activation="sigmoid")
mod3 %>% compile(
  loss="binary_crossentropy",
  optimizer="rmsprop",
  metrics=c("accuracy")
)
summary(mod3)
#
set.seed(57)
checkpoint=callback_model_checkpoint(
  filepath="mod3_AutoEncoderPrev.hdf5",
  save_best_only=TRUE,
  period=1,
  verbose=1
)
early_stopping=callback_early_stopping(patience=5)
#mod3 %>% fit(
#  x=x_train_1,
#  y=y_train_1,
#  epochs=100,
#  batch_size=32,
#  validation_data=list(x_valid_1,y_valid_1),
#  callbacks=list(checkpoint,early_stopping)
#)
mod3=load_model_hdf5("mod3_AutoEncoderPrev.hdf5",compile=FALSE)
pred_mod3=predict(mod3,x_corte_1)
prev_mod3=predict(mod3,x_teste_1)

### Modelos Tipo 2
## Modelo 4
mod4=glm(Class~.,data=treino_2,family=binomial(link="logit"))
pred_mod4=predict(mod4,newdata=as.data.frame(x_corte_2),type="response")
prev_mod4=predict(mod4,newdata=as.data.frame(x_teste_2),type="response")
## Modelo 5
mod5=glm(Class~.,data=treino_2,family=binomial(link="cloglog"))
pred_mod5=predict(mod5,newdata=as.data.frame(x_corte_2),type="response")
prev_mod5=predict(mod5,newdata=as.data.frame(x_teste_2),type="response")
## Modelo 6
#cv_mod6=cv.glmnet(
#  as.matrix(treino_2 %>% select(-Class)),
#  as.matrix(treino_2 %>% select(Class)),
#  alpha=1,
#  family="binomial",
#  nfolds=7,
#  type.measure="class"
#)
#save(file="cv_mod6.RDa",cv_mod6)
load("cv_mod6.RDa")
mod6=glmnet(
  as.matrix(treino_2 %>% select(-Class)),
  as.matrix(treino_2 %>% select(Class)),
  alpha=1,
  lambda=cv_mod6$lambda.1se,
  family="binomial"
)
pred_mod6=predict(mod6,newx=as.matrix(x_corte_2))
prev_mod6=predict(mod6,newx=as.matrix(x_teste_2))

### Modelos Tipo 3
## Modelo 7
cv_mod7=cv.glmnet(
  as.matrix(treino_3 %>% select(-VR)),
  as.matrix(treino_3 %>% select(VR)),
  alpha=1,
  parallel=TRUE,
  nfolds=7,
  type.measure="mae"
)
mod7=glmnet(
  as.matrix(treino_3 %>% select(-VR)),
  as.matrix(treino_3 %>% select(VR)),
  alpha=1,
  lambda=cv_mod7$lambda.1se
)
pred_mod7=predict(mod7,newx=as.matrix(x_corte_3))
prev_mod7=predict(mod7,newx=as.matrix(x_teste_3))
## Modelo 8
mod8=glm(VR~.,data=treino_3)
pred_mod8=predict(mod8,newdata=as.data.frame(x_corte_3))
prev_mod8=predict(mod8,newdata=as.data.frame(x_teste_3))

### Sensibilidade e Especificidade
quantis=seq(from=0,to=1,by=0.005)

## Modelos Tipo 1
# Modelo 1
mod1_k=as.vector(quantile(x=sse_mod1,probs=quantis))
#
mod1_sens=sapply(mod1_k,function(k){
  predicted_class=as.numeric(sse_mod1 > k)
  sum(predicted_class == 1 & y_corte_1 == 1)/sum(y_corte_1 == 1)
})
mod1_esp=sapply(mod1_k,function(k){
  predicted_class=as.numeric(sse_mod1 > k)
  sum(predicted_class == 0 & y_corte_1 == 0)/sum(y_corte_1 == 0)
})
# Modelo 2
mod2_k=as.vector(quantile(x=pred_mod2,probs=quantis))
#
mod2_sens=sapply(mod2_k,function(k){
  predicted_class=as.numeric(pred_mod2 > k)
  sum(predicted_class == 1 & y_corte_1 == 1)/sum(y_corte_1 == 1)
})
mod2_esp=sapply(mod2_k,function(k){
  predicted_class=as.numeric(pred_mod2 > k)
  sum(predicted_class == 0 & y_corte_1 == 0)/sum(y_corte_1 == 0)
})
# Modelo 3
mod3_k=as.vector(quantile(x=pred_mod3,probs=quantis))
#
mod3_sens=sapply(mod3_k,function(k){
  predicted_class=as.numeric(pred_mod3 > k)
  sum(predicted_class == 1 & y_corte_1 == 1)/sum(y_corte_1 == 1)
})
mod3_esp=sapply(mod3_k,function(k){
  predicted_class=as.numeric(pred_mod3 > k)
  sum(predicted_class == 0 & y_corte_1 == 0)/sum(y_corte_1 == 0)
})

## Modelos Tipo 2
# Modelo 4
mod4_k=as.vector(quantile(x=pred_mod4,probs=quantis))
#
mod4_sens=sapply(mod4_k,function(k){
  predicted_class=as.numeric(pred_mod4 > k)
  sum(predicted_class == 1 & y_corte_2 == 1)/sum(y_corte_2 == 1)
})
mod4_esp=sapply(mod4_k,function(k){
  predicted_class=as.numeric(pred_mod4 > k)
  sum(predicted_class == 0 & y_corte_2 == 0)/sum(y_corte_2 == 0)
})
# Modelo 5
mod5_k=as.vector(quantile(x=pred_mod5,probs=quantis))
#
mod5_sens=sapply(mod5_k,function(k){
  predicted_class=as.numeric(pred_mod5 > k)
  sum(predicted_class == 1 & y_corte_2 == 1)/sum(y_corte_2 == 1)
})
mod5_esp=sapply(mod5_k,function(k){
  predicted_class=as.numeric(pred_mod5 > k)
  sum(predicted_class == 0 & y_corte_2 == 0)/sum(y_corte_2 == 0)
})
# Modelo 6
mod6_k=as.vector(quantile(x=pred_mod6,probs=quantis))
#
mod6_sens=sapply(mod6_k,function(k){
  predicted_class=as.numeric(pred_mod6 > k)
  sum(predicted_class == 1 & y_corte_2 == 1)/sum(y_corte_2 == 1)
})
mod6_esp=sapply(mod6_k,function(k){
  predicted_class=as.numeric(pred_mod6 > k)
  sum(predicted_class == 0 & y_corte_2 == 0)/sum(y_corte_2 == 0)
})

## Modelos Tipo 3
# Modelo 7
mod7_k=as.vector(quantile(x=pred_mod7,probs=quantis))
#
mod7_sens=sapply(mod7_k,function(k){
  predicted_class=as.numeric(pred_mod7 < k)
  sum(predicted_class == 1 & y_corte_3 == 1)/sum(y_corte_3 == 1)
})
mod7_esp=sapply(mod7_k,function(k){
  predicted_class=as.numeric(pred_mod7 < k)
  sum(predicted_class == 0 & y_corte_3 == 0)/sum(y_corte_3 == 0)
})
# Modelo 8
mod8_k=as.vector(quantile(x=pred_mod8,probs=quantis))
#
mod8_sens=sapply(mod8_k,function(k){
  predicted_class=as.numeric(pred_mod8 < k)
  sum(predicted_class == 1 & y_corte_3 == 1)/sum(y_corte_3 == 1)
})
mod8_esp=sapply(mod8_k,function(k){
  predicted_class=as.numeric(pred_mod8 < k)
  sum(predicted_class == 0 & y_corte_3 == 0)/sum(y_corte_3 == 0)
})

## Construção dos Gráficos de Desempenho
# Caso 1
qt=0.95
HR_c1=c(
  mean(y_teste_1[sse_mod1_teste > as.vector(quantile(sse_mod1,qt))] == 1),
  mean(y_teste_1[prev_mod2 > quantile(pred_mod2,qt)]==1),
  mean(y_teste_1[prev_mod3 > quantile(pred_mod3,qt)]==1),
  mean(y_teste_2[prev_mod4 > quantile(pred_mod4,qt),]==1),
  #mean(y_teste_2[prev_mod5 > quantile(pred_mod5,qt),]==1),
  #mean(y_teste_2[prev_mod6 > quantile(pred_mod6,qt),]==1),
  mean(y_teste_3[prev_mod7 < quantile(pred_mod7,1-qt)]==1),
  mean(y_teste_3[prev_mod8 < quantile(pred_mod8,1-qt)]==1)
)
PEN_c1=c(
  length(y_teste_1[y_teste_1==1 & sse_mod1_teste > quantile(sse_mod1,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_1[y_teste_1==1 & prev_mod2 > quantile(pred_mod2,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_1[y_teste_1==1 & prev_mod3 > quantile(pred_mod3,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_2[y_teste_2==1 & prev_mod4 > quantile(pred_mod4,qt),])/length(y_teste_2[y_teste_2==1]),
  #length(y_teste_2[y_teste_2==1 & prev_mod5 > quantile(pred_mod5,qt),])/length(y_teste_2[y_teste_2==1]),
  #length(y_teste_2[y_teste_2==1 & prev_mod6 > quantile(pred_mod6,qt),])/length(y_teste_2[y_teste_2==1]),
  length(y_teste_3[y_teste_3==1 & prev_mod7 < quantile(pred_mod7,1-qt)])/length(y_teste_3[y_teste_3==1]),
  length(y_teste_3[y_teste_3==1 & prev_mod8 < quantile(pred_mod8,1-qt)])/length(y_teste_3[y_teste_3==1])
)
ValPrev_c1=c(
  sum(df_teste$Amount[y_teste_1==1 & sse_mod1_teste > quantile(sse_mod1,qt)]),
  sum(df_teste$Amount[y_teste_1==1 & prev_mod2 > quantile(pred_mod2,qt)]),
  sum(df_teste$Amount[y_teste_1==1 & prev_mod3 > quantile(pred_mod3,qt)]),
  sum(df_teste$Amount[y_teste_2==1 & prev_mod4 > quantile(pred_mod4,qt)]),
  #sum(df_teste$Amount[y_teste_2==1 & prev_mod5 > quantile(pred_mod5,qt)]),
  #sum(df_teste$Amount[y_teste_2==1 & prev_mod6 > quantile(pred_mod6,qt)]),
  sum(df_teste$Amount[y_teste_3==1 & prev_mod7 < quantile(pred_mod7,1-qt)]),
  sum(df_teste$Amount[y_teste_3==1 & prev_mod8 < quantile(pred_mod8,1-qt)])
)
# Caso 2
qt=0.98
HR_c2=c(
  mean(y_teste_1[sse_mod1_teste > as.vector(quantile(sse_mod1,qt))] == 1),
  mean(y_teste_1[prev_mod2 > quantile(pred_mod2,qt)]==1),
  mean(y_teste_1[prev_mod3 > quantile(pred_mod3,qt)]==1),
  mean(y_teste_2[prev_mod4 > quantile(pred_mod4,qt),]==1),
  #mean(y_teste_2[prev_mod5 > quantile(pred_mod5,qt),]==1),
  #mean(y_teste_2[prev_mod6 > quantile(pred_mod6,qt),]==1),
  mean(y_teste_3[prev_mod7 < quantile(pred_mod7,1-qt)]==1),
  mean(y_teste_3[prev_mod8 < quantile(pred_mod8,1-qt)]==1)
)
PEN_c2=c(
  length(y_teste_1[y_teste_1==1 & sse_mod1_teste > quantile(sse_mod1,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_1[y_teste_1==1 & prev_mod2 > quantile(pred_mod2,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_1[y_teste_1==1 & prev_mod3 > quantile(pred_mod3,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_2[y_teste_2==1 & prev_mod4 > quantile(pred_mod4,qt),])/length(y_teste_2[y_teste_2==1]),
  #length(y_teste_2[y_teste_2==1 & prev_mod5 > quantile(pred_mod5,qt),])/length(y_teste_2[y_teste_2==1]),
  #length(y_teste_2[y_teste_2==1 & prev_mod6 > quantile(pred_mod6,qt),])/length(y_teste_2[y_teste_2==1]),
  length(y_teste_3[y_teste_3==1 & prev_mod7 < quantile(pred_mod7,1-qt)])/length(y_teste_3[y_teste_3==1]),
  length(y_teste_3[y_teste_3==1 & prev_mod8 < quantile(pred_mod8,1-qt)])/length(y_teste_3[y_teste_3==1])
)
ValPrev_c2=c(
  sum(df_teste$Amount[y_teste_1==1 & sse_mod1_teste > quantile(sse_mod1,qt)]),
  sum(df_teste$Amount[y_teste_1==1 & prev_mod2 > quantile(pred_mod2,qt)]),
  sum(df_teste$Amount[y_teste_1==1 & prev_mod3 > quantile(pred_mod3,qt)]),
  sum(df_teste$Amount[y_teste_2==1 & prev_mod4 > quantile(pred_mod4,qt)]),
  #sum(df_teste$Amount[y_teste_2==1 & prev_mod5 > quantile(pred_mod5,qt)]),
  #sum(df_teste$Amount[y_teste_2==1 & prev_mod6 > quantile(pred_mod6,qt)]),
  sum(df_teste$Amount[y_teste_3==1 & prev_mod7 < quantile(pred_mod7,1-qt)]),
  sum(df_teste$Amount[y_teste_3==1 & prev_mod8 < quantile(pred_mod8,1-qt)])
)
# Caso 3
qt=0.99
HR_c3=c(
  mean(y_teste_1[sse_mod1_teste > as.vector(quantile(sse_mod1,qt))]==1),
  mean(y_teste_1[prev_mod2 > quantile(pred_mod2,qt)]==1),
  mean(y_teste_1[prev_mod3 > quantile(pred_mod3,qt)]==1),
  mean(y_teste_2[prev_mod4 > quantile(pred_mod4,qt),]==1),
  #mean(y_teste_2[prev_mod5 > quantile(pred_mod5,qt),]==1),
  #mean(y_teste_2[prev_mod6 > quantile(pred_mod6,qt),]==1),
  mean(y_teste_3[prev_mod7 < quantile(pred_mod7,1-qt)]==1),
  mean(y_teste_3[prev_mod8 < quantile(pred_mod8,1-qt)]==1)
)
PEN_c3=c(
  length(y_teste_1[y_teste_1==1 & sse_mod1_teste > quantile(sse_mod1,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_1[y_teste_1==1 & prev_mod2 > quantile(pred_mod2,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_1[y_teste_1==1 & prev_mod3 > quantile(pred_mod3,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_2[y_teste_2==1 & prev_mod4 > quantile(pred_mod4,qt),])/length(y_teste_2[y_teste_2==1]),
  #length(y_teste_2[y_teste_2==1 & prev_mod5 > quantile(pred_mod5,qt),])/length(y_teste_2[y_teste_2==1]),
  #length(y_teste_2[y_teste_2==1 & prev_mod6 > quantile(pred_mod6,qt),])/length(y_teste_2[y_teste_2==1]),
  length(y_teste_3[y_teste_3==1 & prev_mod7 < quantile(pred_mod7,1-qt)])/length(y_teste_3[y_teste_3==1]),
  length(y_teste_3[y_teste_3==1 & prev_mod8 < quantile(pred_mod8,1-qt)])/length(y_teste_3[y_teste_3==1])
)
ValPrev_c3=c(
  sum(df_teste$Amount[y_teste_1==1 & sse_mod1_teste > quantile(sse_mod1,qt)]),
  sum(df_teste$Amount[y_teste_1==1 & prev_mod2 > quantile(pred_mod2,qt)]),
  sum(df_teste$Amount[y_teste_1==1 & prev_mod3 > quantile(pred_mod3,qt)]),
  sum(df_teste$Amount[y_teste_2==1 & prev_mod4 > quantile(pred_mod4,qt)]),
  #sum(df_teste$Amount[y_teste_2==1 & prev_mod5 > quantile(pred_mod5,qt)]),
  #sum(df_teste$Amount[y_teste_2==1 & prev_mod6 > quantile(pred_mod6,qt)]),
  sum(df_teste$Amount[y_teste_3==1 & prev_mod7 < quantile(pred_mod7,1-qt)]),
  sum(df_teste$Amount[y_teste_3==1 & prev_mod8 < quantile(pred_mod8,1-qt)])
)
# Caso 4
qt=0.995
HR_c4=c(
  mean(y_teste_1[sse_mod1_teste > as.vector(quantile(sse_mod1,qt))] == 1),
  mean(y_teste_1[prev_mod2 > quantile(pred_mod2,qt)]==1),
  mean(y_teste_1[prev_mod3 > quantile(pred_mod3,qt)]==1),
  mean(y_teste_2[prev_mod4 > quantile(pred_mod4,qt),]==1),
  #mean(y_teste_2[prev_mod5 > quantile(pred_mod5,qt),]==1),
  #mean(y_teste_2[prev_mod6 > quantile(pred_mod6,qt),]==1),
  mean(y_teste_3[prev_mod7 < quantile(pred_mod7,1-qt)]==1),
  mean(y_teste_3[prev_mod8 < quantile(pred_mod8,1-qt)]==1)
)
PEN_c4=c(
  length(y_teste_1[y_teste_1==1 & sse_mod1_teste > quantile(sse_mod1,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_1[y_teste_1==1 & prev_mod2 > quantile(pred_mod2,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_1[y_teste_1==1 & prev_mod3 > quantile(pred_mod3,qt)])/length(y_teste_1[y_teste_1==1]),
  length(y_teste_2[y_teste_2==1 & prev_mod4 > quantile(pred_mod4,qt),])/length(y_teste_2[y_teste_2==1]),
  #length(y_teste_2[y_teste_2==1 & prev_mod5 > quantile(pred_mod5,qt),])/length(y_teste_2[y_teste_2==1]),
  #length(y_teste_2[y_teste_2==1 & prev_mod6 > quantile(pred_mod6,qt),])/length(y_teste_2[y_teste_2==1]),
  length(y_teste_3[y_teste_3==1 & prev_mod7 < quantile(pred_mod7,1-qt)])/length(y_teste_3[y_teste_3==1]),
  length(y_teste_3[y_teste_3==1 & prev_mod8 < quantile(pred_mod8,1-qt)])/length(y_teste_3[y_teste_3==1])
)
ValPrev_c4=c(
  sum(df_teste$Amount[y_teste_1==1 & sse_mod1_teste > quantile(sse_mod1,qt)]),
  sum(df_teste$Amount[y_teste_1==1 & prev_mod2 > quantile(pred_mod2,qt)]),
  sum(df_teste$Amount[y_teste_1==1 & prev_mod3 > quantile(pred_mod3,qt)]),
  sum(df_teste$Amount[y_teste_2==1 & prev_mod4 > quantile(pred_mod4,qt)]),
  #sum(df_teste$Amount[y_teste_2==1 & prev_mod5 > quantile(pred_mod5,qt)]),
  #sum(df_teste$Amount[y_teste_2==1 & prev_mod6 > quantile(pred_mod6,qt)]),
  sum(df_teste$Amount[y_teste_3==1 & prev_mod7 < quantile(pred_mod7,1-qt)]),
  sum(df_teste$Amount[y_teste_3==1 & prev_mod8 < quantile(pred_mod8,1-qt)])
)
df_desempenho=data.frame(
  "Value"=c(HR_c1,HR_c2,HR_c3,HR_c4,
            PEN_c1,PEN_c2,PEN_c3,PEN_c4,
            ValPrev_c1,ValPrev_c2,ValPrev_c3,ValPrev_c4),
  "Caso"=rep(rep(c("5%","2%","1%","0.5%"),each=6),3),
  "Medida"=rep(c("Hit Rate","Penetração","Valor Prevenido"),each=4*6),
  "Modelo"=rep(rep(c("AE_Pred","AE_Bin","RNA_Bin","Reg_Log_C","Las_Lin","Reg_Lin_C"),times=4),times=3) %>% as.character
)

pdf(file="/home/mauricio/Documents/Faculdade/Semestres/2018.1/Trabalho de Graduação 1/GraficosTG.pdf",width=7,height=5)
# G1 - Visualização das Covariáveis
bb=aa[,1:30]
for(i in 1:ncol(bb)){
  bb[,i]=percent_rank(aa[,i])
}
bb$"Tipo de Transação"=ifelse(aa$Class==1,"Fraudulenta","Não Fraudulenta")
names(bb)[c(1,30)]=c("Tempo","Valor")
bb %>% gather(variaveis,value,-`Tipo de Transação`) %>%
  ggplot(aes(y=factor(variaveis,levels=c(paste("V",28:1,sep=""),"Tempo","Valor")),
             fill=`Tipo de Transação`,
             x=value))+
  geom_density_ridges()+
  xlab("Valores Padronizados")+
  ylab("Distribuição")+
  theme_minimal()+
  theme(text=element_text(size=17))
# G2 - Distribuição do Score por Classe mod1
df1_corte=data.frame("ErroModelo"=sse_mod1,
                     "Classe"=ifelse(y_corte_1==1,"Fraudulenta","Não Fraudulenta"))
ggplot(df1_corte,aes(x=ErroModelo,fill=Classe))+
  geom_density(n=8192)+
  ylim(0,5)+
  ylab("Densidade Estimada")+
  xlab("Erro de Previsão")+
  theme_minimal()+
  theme(text=element_text(size=20))
# G3 - Distribuição do Score por Classe mod2
df2_corte=data.frame("ErroModelo"=pred_mod2,
                     "Classe"=ifelse(y_corte_1==1,"Fraudulenta","Não Fraudulenta"))
ggplot(df2_corte,aes(x=ErroModelo,fill=Classe,col=Classe))+
  geom_density()+
  ylim(0,10)+
  ylab("Densidade Estimada")+
  xlab("Valor Predito")+
  theme_minimal()+
  theme(text=element_text(size=20))
# G4 - Distribuição do Score por Classe mod3
df3_corte=data.frame("ErroModelo"=pred_mod3,
                     "Classe"=ifelse(y_corte_1==1,"Fraudulenta","Não Fraudulenta"))
ggplot(df3_corte,aes(x=ErroModelo,fill=Classe,col=Classe))+
  geom_density()+
  ylim(0,10)+
  ylab("Densidade Estimada")+
  xlab("Valor de Predito")+
  theme_minimal()+
  theme(text=element_text(size=20))
## G5 - Curvas ROC
# Construção do conjunto de dados único
df2=data.frame("Sensibilidade"=c(mod1_sens,mod2_sens,mod3_sens,mod4_sens,mod5_sens,mod6_sens,mod7_sens,mod8_sens),
               "Especificidade"=c(mod1_esp,mod2_esp,mod3_esp,mod4_esp,mod5_esp,mod6_esp,mod7_esp,mod8_esp),
               "Modelo"=rep(c("AE_Pred","AE_Bin","RNA_Bin","Reg_Log_C","Reg_CLogLog","Las_Log","Las_Lin","Reg_Lin_C"),each=length(quantis)),
               "Possíveis K's"=c(mod1_k,mod2_k,mod3_k,mod4_k,mod5_k,mod6_k,mod7_k,mod8_k))
ggplot(df2,aes(x=1-Especificidade,y=Sensibilidade,col=Modelo))+
  geom_line(size=1.1)+
  geom_segment(x=0,y=0,xend=1,yend=1,col="black")+
  theme_minimal()
## G6 - Comparando Penetração
df_desempenho %>%
  filter(Medida=="Penetração") %>%
  ggplot(aes(x=Caso,y=Value,fill=Modelo))+
  geom_bar(stat="identity",position=position_dodge())+
  xlab("% de Negadas")+
  ylab("Penetração")+
  ylim(0,1)+
  theme_minimal()+
  theme(text=element_text(size=18))
## G7 - Comparando Hit Rate
df_desempenho %>%
  filter(Medida=="Hit Rate") %>%
  ggplot(aes(x=Caso,y=Value,fill=Modelo))+
  geom_bar(stat="identity",position=position_dodge())+
  xlab("% de Negadas")+
  ylab("Hit Rate")+
  ylim(0,0.4)+
  theme_minimal()+
  theme(text=element_text(size=18))
## G8 - Comparando Valor Prevenido
df_desempenho %>%
  filter(Medida=="Valor Prevenido") %>%
  ggplot(aes(x=Caso,y=Value,fill=Modelo))+
  geom_bar(stat="identity",position=position_dodge())+
  xlab("% de Negadas")+
  ylab("Valor de Fraude Prevenida ($)")+
  theme_minimal()+
  theme(text=element_text(size=18))
dev.off()