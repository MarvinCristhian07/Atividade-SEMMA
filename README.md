# Previsão de Churn - Framework SEMMA

Este projeto aplica o framework SEMMA (Sample, Explore, Modify, Model, Assess) para desenvolver um modelo preditivo de identificação de churn (evasão), utilizando dados da Analytical Base Table (ABT) Churn Journey.

## 🚀 Estrutura do Projeto

- `abt_churn.csv`: Base de dados utilizada.
- `churn_semma.py`: Script principal de mineração e modelagem.
- `modelo_churn_pipeline.pkl`: Pipeline completo exportado (preprocessamento + modelo).

## 🛠️ Metodologia (SEMMA)

1. **Sample**: Separação de base Out-of-Time (OOT) da safra 2025-04 para validação de estabilidade temporal. Divisão de treino/teste com estratificação do target.
2. **Explore**: Análise bivariada e cálculo de relevância das variáveis explicativas frente ao Churn.
3. **Modify**: Construção de pipeline automatizado com `KBinsDiscretizer` para variáveis contínuas e `OneHotEncoder` para categóricas.
4. **Model**: Implementação de `RandomForestClassifier` com otimização de hiperparâmetros via `RandomizedSearchCV` e validação cruzada.
5. **Assess**: Avaliação por ROC AUC e análise de Gains/Lift, garantindo que o modelo capture a maioria dos churners nos decis de maior risco.

## 💻 Como executar

```bash
# Instalar dependências
pip install pandas scikit-learn matplotlib

# Executar o pipeline
python churn_semma.py
