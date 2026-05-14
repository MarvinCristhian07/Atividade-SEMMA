# Exercicio Pratico: Previsao de Churn com Framework SEMMA
# Autor: Heitor Vitti Partezani
# Base de Dados: Analytical Base Table Churn (Kaggle - Teo Calvo)

# Importacoes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# Carregamento dos Dados
df = pd.read_csv("abt_churn.csv")
print("Shape:", df.shape)

# Definir target e features (removendo colunas de identificacao)
target = "flagChurn"
features = [c for c in df.columns if c not in ["dtRef", "idUsuario", "flagChurn"]]

# 1. S - SAMPLE (Amostragem)
print("\n--- 1. SAMPLE ---")

# 1.1 Separar a ultima safra como Out-of-Time (OOT)
# A base OOT nao sera tocada ate o final - testa estabilidade temporal
ultima_safra = sorted(df["dtRef"].unique())[-1]  # 2025-04-01
df_oot = df[df["dtRef"] == ultima_safra].copy()
df_dev = df[df["dtRef"] != ultima_safra].copy()

# 1.2 Dividir o restante em Treino (80%) e Teste (20%) com estratificacao
# stratify=y garante que a proporcao de churn seja igual em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    df_dev[features], df_dev[target],
    test_size=0.2, random_state=42, stratify=df_dev[target]
)
X_oot = df_oot[features]
y_oot = df_oot[target]

print(f"Treino: {len(X_train)} | Churn rate: {y_train.mean():.4f}")
print(f"Teste:  {len(X_test)}  | Churn rate: {y_test.mean():.4f}")
print(f"OOT:    {len(X_oot)}   | Churn rate: {y_oot.mean():.4f}")

# 2. E - EXPLORE (Exploracao) - somente na base de Treino
print("\n--- 2. EXPLORE ---")

# 2.1 Verificacao de valores nulos
print(f"Valores nulos no treino: {X_train.isnull().sum().sum()}")
# Resultado: 0 nulos.

# 2.2 Analise Bivariada: media das features separando Churn=0 vs Churn=1
df_analise = X_train.copy()
df_analise["flagChurn"] = y_train.values
bivariada = df_analise.groupby("flagChurn").mean().T
bivariada.columns = ["Nao Churn", "Churn"]
bivariada["Diff%"] = ((bivariada["Churn"] - bivariada["Nao Churn"])
                      / bivariada["Nao Churn"].replace(0, np.nan) * 100).round(1)
print("\nTop 10 features com maior diferenca entre grupos:")
print(bivariada.sort_values("Diff%", key=abs, ascending=False).head(10).to_string())

# Grafico da analise bivariada
top10 = bivariada["Diff%"].abs().sort_values(ascending=True).tail(10)
top10.plot(kind="barh", figsize=(9, 4), title="Top 10 Features - Diferenca Churn vs Nao-Churn (%)")
plt.xlabel("Diferenca relativa (%)")
plt.tight_layout()
plt.savefig("02_analise_bivariada.png", dpi=150)
plt.close()

# 2.3 Feature Importance com Arvore de Decisao simples
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

importancias = pd.Series(tree.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance (Arvore de Decisao):")
print(importancias.head(10).to_string())

# Grafico feature importance
importancias.head(10).sort_values().plot(kind="barh", figsize=(9, 4),
                                         title="Feature Importance - Arvore de Decisao")
plt.xlabel("Importancia")
plt.tight_layout()
plt.savefig("02_feature_importance.png", dpi=150)
plt.close()

# 3. M - MODIFY (Modificacao)
print("\n--- 3. MODIFY ---")

# 3.1 Discretizacao: transforma as 5 features mais importantes em faixas (bins)
# 3.2 One-Hot Encoding: transforma essas faixas em colunas binarias
# 3.3 Tudo dentro de um Pipeline para nao repetir codigo no teste/OOT

top5 = importancias.head(5).index.tolist()  # 5 features mais importantes
demais = [f for f in features if f not in top5]  # restante passa direto

preprocessor = ColumnTransformer([
    ("bin_ohe", Pipeline([
        ("bin", KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")),
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ]), top5),
    ("num", "passthrough", demais),
])

print(f"Features discretizadas: {top5}")
print(f"Features passthrough:   {len(demais)}")

# 4. M - MODEL (Modelagem)
print("\n--- 4. MODEL ---")

# Pipeline completo: preprocessamento + classificador
pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=42)),
])

# Otimizacao de hiperparametros com RandomizedSearchCV + validacao cruzada
params = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [5, 10, 15, 20, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Otimizando hiperparametros (RandomizedSearchCV)...")
busca = RandomizedSearchCV(pipe, params, n_iter=20, cv=cv,
                           scoring="roc_auc", random_state=42, n_jobs=-1)
busca.fit(X_train, y_train)

modelo = busca.best_estimator_
print(f"Melhor AUC (CV): {busca.best_score_:.4f}")
print(f"Melhores params: {busca.best_params_}")

# 5. A - ASSESS (Avaliacao)
print("\n--- 5. ASSESS ---")

# 5.1 Metricas: ROC AUC e Acuracia em Treino, Teste e OOT
for nome, X, y in [("Treino", X_train, y_train),
                    ("Teste", X_test, y_test),
                    ("OOT", X_oot, y_oot)]:
    prob = modelo.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)
    acc = accuracy_score(y, modelo.predict(X))
    print(f"  {nome:6s} -> AUC: {auc:.4f} | Acuracia: {acc:.4f}")

# Analise de overfitting: se AUC do treino for muito maior que teste = overfitting
auc_treino = roc_auc_score(y_train, modelo.predict_proba(X_train)[:, 1])
auc_teste = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
print(f"\n  Diff AUC Treino-Teste: {auc_treino - auc_teste:.4f} (< 0.05 = OK)")

# Curva ROC
fig, ax = plt.subplots(figsize=(7, 5))
for nome, X, y in [("Treino", X_train, y_train),
                    ("Teste", X_test, y_test),
                    ("OOT", X_oot, y_oot)]:
    prob = modelo.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, prob)
    ax.plot(fpr, tpr, label=f"{nome} (AUC={roc_auc_score(y, prob):.4f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatorio")
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("Curva ROC")
ax.legend()
plt.tight_layout()
plt.savefig("05_curva_roc.png", dpi=150)
plt.close()

# 5.2 Analise de Negocio (Lift/Gains)
# Ordena os clientes pela probabilidade de churn e mostra quantos churners
# sao capturados se abordamos apenas os top 10%, 20%, 30%
prob_teste = modelo.predict_proba(X_test)[:, 1]
df_lift = pd.DataFrame({"real": y_test.values, "prob": prob_teste})
df_lift = df_lift.sort_values("prob", ascending=False).reset_index(drop=True)
total_churn = df_lift["real"].sum()

print("\n  Lift/Gains (base de Teste):")
print(f"  {'Top%':>5} | {'Captura%':>9} | {'Lift':>5}")
print("  " + "-" * 27)
gains_x, gains_y = [0], [0]
for pct in [10, 20, 30, 40, 50]:
    n = int(len(df_lift) * pct / 100)
    captura = df_lift["real"].iloc[:n].sum() / total_churn * 100
    print(f"  {pct:4d}% | {captura:8.1f}% | {captura/pct:.2f}x")
    gains_x.append(pct)
    gains_y.append(captura)
gains_x.append(100)
gains_y.append(100)

# Grafico de Gains
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(gains_x, gains_y, "o-", label="Modelo")
ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="Aleatorio")
ax.set_xlabel("% da base abordada")
ax.set_ylabel("% de Churners capturados")
ax.set_title("Curva de Gains - Modelo vs Aleatorio")
ax.legend()
plt.tight_layout()
plt.savefig("05_curva_gains.png", dpi=150)
plt.close()

# 5.3 Serializacao: salvar o modelo treinado em arquivo .pkl
with open("modelo_churn.pkl", "wb") as f:
    pickle.dump(modelo, f)
print("\n  Modelo salvo em: modelo_churn.pkl")

print("\n=== Exercicio SEMMA concluido! ===")
