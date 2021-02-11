
# Carregando os pacotes
library(forecast)
library(ggplot2)
library(seasonal)
library(seasonalview)
library(urca) # unit root test
library(tseries) # outro teste de estacionariedade
library(lmtest) # teste alternativo ACF
library(zoo)

# Carregando o TS Nottem
data(nottem)


# Verificando o tipo de Objeto
class(nottem)


# Visualizando o Dataset
View(nottem)


### Data Preparation / Quality ###

plot(nottem) #Visualiza o TS. Aparentemente Sazonal, estacionária e sem Outliers
boxplot(nottem) #não é notado presença de Outliers no dataset


#Detecção automática de Outliers
mytsOut = tsoutliers(nottem)
mytsOut #Nenhum Outlier detectado nesta TS através do método automático
plot(nottem) #Caso houvesse remoção de Outliers, plot novamente para visualizar o TS sem os mesmos


# Cleaning NA and outliers with forecast package (Não necessário pois a base não apresenta NA quando visualizamos
# o TS via view())
mytsclean = tsclean(nottem)
plot(mytsclean)
summary(mytsclean)
# Sem outliers


### EXPLORE E VISUALIZE SERIES ###


# Visualiza o dataset
datasets::nottem #neste caso será igual ao plot anterior, pois não houve remoção de NA e de Outliers
#ou
print(nottem)


# Criando a nova visualização TS (se houve remoção de Outliers e NA)
plot(nottem)
?nottem

# Análise Exploratoria do dataset
min(nottem)
max(nottem)
mean(nottem)
median(nottem)
summary(nottem) #dados acima podem ser resumidos pelo comando summary
length(nottem)
start(nottem)
end(nottem)
frequency(nottem)
cycle(nottem)
boxplot(nottem) #não é notado presença de Outliers no dataset
hist(nottem) #Dados não apresentam dist normal. A ser confirmado através do Shapiro Test


#teste de normalidade
shapiro.test(nottem) # p-value 1.32 (bem superior ao 0,05), confirmando a não distribuição dos dados


# Teste ACF - Correlograma (Análise de Correlação)
acf(nottem, na.action = na.pass ) #confirma a correlação dos dados (limites azuis rompidos) através do ACF
#e aparentemente com sazonalidade, o que será confirmado através da decomposição da série temporal.
#ACF 3/20 abaixo/acima, significa que os residuais ainda tem informação e o fcst pode ser melhorado.
tsdisplay(nottem)
# teste acf alternatido do pacote lmtest. 240 é o length do dataset
dwtest(nottem[-240] ~ nottem[-1])# resultado confirmando correlação



# Decomposição do Time Series
decompose(nottem) #para visualizar os dados numéricos no console
x<- decompose(nottem)
plot(x) #plotar a decomposição do time series
#É possível notar que o ts aparentemente apresenta tendencia (a ser confirmado pelo teste de estacionariedade), 
#sazonalidade e correlação (random) dos dados. Com isso, se faz interessante, testar o Seasonal Arima como o exponencial
#método de decomposição alternativo do pacote
plot(stl(nottem, s.window="periodic"))
stl(nottem, s.window="periodic")


# Analisando a tendencia a parte
autoplot(x$trend)


# Verificando o padrão da sazonalidade
ggseasonplot(nottem)


#Verificando o padrão dentro do mês
ggmonthplot(nottem)


# Teste de estacionariedade
x <- ur.kpss(nottem) #Roda o teste
print(x) #resultado de 0.0321, portanto, menor que 0.05 no teste de estacionariedade mostrou que a ts é "sim" 
#estacionária e portanto, não necessitando de transformação "ndiffs(nottem)" para realizar o método ARIMA ou Seas ARIMA
adf.test(nottem) # outro teste de estacionariedade, porém este, do pacote tseries



###  PARTITION SERIES   ###


# Criando dataset para Treino e Teste
DSTreino = ts(nottem, start = c(1920,1), end = c(1937,12), frequency=12)# Dataset onde iremos treinar nosso método
# de Forecast e comparar o resultado com o dataset de Teste (Dados reais)
DSTeste  = ts(nottem, start = c(1938,1), end = c(1939,12), frequency=12) # Base onde o modelo de Treino será comarado
# com o Teste (Dataset com os dados reais, dados que já aconteceram)



### APPLY FORECASTING METHODS ###


## Benchmark Methods ##

meanm <- meanf(DSTreino, h=24)
naivem <- naive(DSTreino, h=24)
driftm <- rwf(DSTreino, h=24, drift = T)


# Gerando o resultado 24m meses para frente
fcstmeanm = forecast(meanm, h=24)
fcstnaivem = forecast(naivem, h=24)
fcstdriftm = forecast(driftm, h=24)


# Visualizando o resultado do modelo
print(fcstmeanm)
print(fcstnaivem)
print(fcstdriftm)


# Gerando gráfico da previsão
plot(meanm, main = "")
lines(naivem$mean, col=123, lwd = 2)
lines(driftm$mean, col=22, lwd = 2)
legend("topleft",lty=1,col=c(4,123,22),
       legend=c("Mean method","Naive method","Drift Method"))


## Modelo ARIMA ##

# Criando Modelo ARIMA Dataset de Treino
ModeloArima = auto.arima(DSTreino,  trace = T,stepwise = F, approximation = F )
#Resultado gerado do índice de performance foi "Best model: ARIMA(1,0,0)(2,1,1)[12]", ou seja, 1,0,0 para a parte 
#não sazonal e 2,1,1 para a parte sazonal e frequencia 12.
# Leg: (1,2,3), onde 1= Autoregressão, 2= diferenciação , 3= moving average


# Gerando o resultado 24m meses para frente
fcstArima = forecast(ModeloArima, h=24)


# Visualizando o resultado do modelo
print(fcstArima)


# Gerando gráfico da previsão
plot(fcstArima)


# Visualizando os residuos do modelo Arima
checkresiduals(ModeloArima) # Ljung-Box test
#no resultado obtido p-value 0.5298 (bem superior ao 0.05 esperado para existência de uma correlação), podemos notar 
#que o mesmo não apresentou correlação (o que é ótimo = "White Noise) e os dados apresentaram uma distribuição normal
#(o que é bom e já esperado para os resíduos). Sendo assim, podemos considerar que o resultado do modelo preditivo 
#Auto Arima foi bom e podemos prosseguir para o próximo modelo para comparar ambos.


# Shappiro Test nos residuais
shapiro.test(ModeloArima$residuals) 
#Para confirmar a distribuição normal dos resíduos (o que [e esperado), realizamos o Shappiro Test nos residuais,
#pudemos confirmar #novamente, com p-value e 0.3491 (bem superior ao 0,05), e assim, 
#confirmando a distribuição normal dos dados.


# Checando a variancia e média dos resíduos
var(ModeloArima$residuals) # espera-se próximo de 1
mean(ModeloArima$residuals) # espera-se próximo de 0
#Ambos apresentaram valores baixos, média bem próxima de zero, o que indica um bom modelo.


## Modelo EXPONENTIAL SMOOTH HOLD WINTERS SEASONAL ##

# Criando Modelo e gerando resultado holt winter - sazonal aditivo
fcstHWA = hw(DSTreino,seasonal = "additive", h=24)


# Criando Modelo e gerando resultado holt winter - sazonal multiplicativo
fcstHWM = hw(DSTreino,seasonal = "multiplicative", h=24)


# Visualizando o resultado do modelo
print(fcstHWA)
print(fcstHWM)


# Gerando gráfico da previsão
autoplot(fcstHWA)
autoplot(fcstHWM)


## Modelo EXPONENTIAL SMOOTH ETS ##

# Criando Modelo Exp Smooth ETS de Treino
ModeloES = ets(DSTreino)


# Gerando o resultado 24m meses para frente
fcstES = forecast(ModeloES, h=24)


# Visualizando o resultado do modelo
print(fcstES)


# Gerando gráfico da previsão
plot(fcstES)


# Visualizando os residuos do modelo Exponential
checkresiduals(ModeloES) # Ljung-Box test
#no resultado obtido p-value 1.746 (bem superior ao 0.05 esperado para existência de uma correlação), podemos notar 
#que o mesmo não apresentou correlação (o que é ótimo = "White Noise) e os dados apresentaram uma distribuição normal
#(o que é bom e já esperado para os resíduos). Sendo assim, podemos considerar que o resultado do modelo preditivo 
#Auto Arima foi bom e podemos prosseguir para o próximo modelo para comparar ambos.


# Shappiro Test nos residuais
shapiro.test(ModeloES$residuals) 
#Para confirmar a distribuição normal dos resíduos (o que [e esperado), realizamos o Shappiro Test nos residuais,
#pudemos confirmar #novamente, com p-value e 0.3491 (bem superior ao 0,05), e assim, 
#confirmando a distribuição normal dos dados.


# Checando a variancia e média dos resíduos
var(ModeloES$residuals)
mean(ModeloES$residuals)
#Ambos apresentaram valores baixos, média bem próxima de zero, o que indica um bom modelo.


# Plotando as duas previsões no mesmo gráfico para o Dataset de Teste
plot(fcstArima)
lines(fcstES$mean, col="red")


plot(nottem) #plotando a previsão final no mesmo gráfico sem o intervalo de confiança para o Dataset Nottem
lines(fcstArima$mean, col="blue")
lines(fcstES$mean, col="red")



### EVALUATE E COMPARE ACCURACY ###

accuracy(fcstmeanm, DSTeste)

accuracy(fcstnaivem, DSTeste)

accuracy(fcstdriftm, DSTeste)

accuracy(fcstArima, DSTeste) #comparo meu modelo Arima do dataset de Treino com o dataset de Teste

accuracy(fcstES, DSTeste) #comparo meu modelo Exponential do dataset de Treino com o dataset de Teste

# Resultado indicou que no geral, os indices de performance (Test Set)foram menores para o AUTO.ARIMA, ou seja, 
# meu modelo ARIMA apresentou melhor resultado para este Time Series (Nottem) do que o modelo Exponential



### IMPLEMENT FORECAST ###


# Aplicando o melhor método (Modelo ARIMA) no dataset completo
ModeloFinal = auto.arima(nottem,  trace = T,stepwise = F, approximation = F )
#Resultado gerado do índice de performance foi "Best model: ARIMA(1,0,0)(2,1,1)[12]", ou seja, 1,0,0 para a parte 
#não sazonal e 2,1,1para a parte sazonal e frequencia 12


# Gerando o resultado 24m meses para frente
fcstfinal = forecast(ModeloFinal, h=24)


# Visualizando o resultado do modelo
print(fcstfinal)


# Gerando gráfico da previsão
plot(fcstfinal)







