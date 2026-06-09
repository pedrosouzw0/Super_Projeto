# =============================================================================
# DASHBOARD — ANÁLISE DE RISCO DE CRÉDITO
# Give Me Some Credit | Probabilidade e Classificação
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, roc_curve)

# ─── CONFIGURAÇÃO DA PÁGINA ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise de Risco de Crédito",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── ESTILO CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
    font-size: 2.2rem; font-weight: 700; color: #1F3864;
    border-bottom: 3px solid #2E75B6; padding-bottom: 0.4rem; margin-bottom: 0.2rem;
  }
  .sub-title {
    font-size: 1rem; color: #555; margin-bottom: 1.5rem;
  }
  .section-header {
    font-size: 1.4rem; font-weight: 700; color: #1F3864;
    background: #D6E4F0; padding: 0.5rem 1rem;
    border-left: 5px solid #2E75B6; border-radius: 4px; margin: 1rem 0 0.5rem 0;
  }
  .metric-card {
    background: #F2F7FC; border: 1px solid #BDD7EE;
    border-radius: 8px; padding: 1rem; text-align: center;
  }
  .metric-label { font-size: 0.8rem; color: #666; font-weight: 600; text-transform: uppercase; }
  .metric-value { font-size: 2rem; font-weight: 700; color: #1F3864; }
  .pred-box {
    border-radius: 10px; padding: 1.2rem; text-align: center;
    font-size: 1.1rem; font-weight: 700; margin: 0.3rem 0;
  }
  .pred-inadim { background: #FDECEA; border: 2px solid #E24B4A; color: #791F1F; }
  .pred-adim   { background: #E8F5EE; border: 2px solid #1D9E75; color: #085041; }
  .insight-box {
    background: #F2F7FC; border-left: 4px solid #2E75B6;
    padding: 0.8rem 1rem; border-radius: 0 6px 6px 0;
    font-size: 0.9rem; color: #333; margin: 0.5rem 0;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: #F2F7FC; border-radius: 6px 6px 0 0;
    padding: 0.5rem 1.2rem; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ─── DADOS E MODELOS (cache para não re-treinar a cada interação) ─────────────
@st.cache_data(show_spinner="Carregando dataset...")
def carregar_dados():
    try:
        df = pd.read_csv('dataset_limpo.csv')
        # garante total_atrasos
        if 'total_atrasos' not in df.columns:
            cols_a = [c for c in df.columns if 'PastDue' in c or '90Days' in c]
            df['total_atrasos'] = df[cols_a].sum(axis=1)
        # garante colunas categóricas
        if 'faixa_etaria' not in df.columns:
            df['faixa_etaria'] = pd.cut(df['age'],
                bins=[17,30,45,60,101],
                labels=['Jovem (18-30)','Adulto (31-45)','Maduro (46-60)','Sênior (61+)'])
        if 'faixa_renda' not in df.columns:
            df['faixa_renda'] = pd.cut(df['MonthlyIncome'],
                bins=[-1,2000,5000,10000,99999999],
                labels=['Baixa (<2k)','Média (2k-5k)','Alta (5k-10k)','Muito Alta (>10k)'])
        def perfil(x):
            if x==0: return 'Sem atrasos'
            elif x<=2: return 'Poucos atrasos'
            else: return 'Muitos atrasos'
        if 'perfil_historico' not in df.columns:
            df['perfil_historico'] = df['total_atrasos'].apply(perfil)
        return df
    except FileNotFoundError:
        np.random.seed(42)
        N = 140000
        default = np.random.binomial(1, 0.067, N)
        age = np.random.normal(52, 14, N).clip(18, 95)
        income = np.random.lognormal(np.log(5400), 0.8, N).clip(500, 30000)
        income[default==1] *= 0.78
        revolving = np.random.beta(0.6, 2.5, N)
        revolving[default==1] = np.random.beta(2,1.5,default.sum()).clip(0,1.2)
        debt = np.random.lognormal(np.log(0.35),1.2,N).clip(0,3)
        def atr(n,t):
            b=np.zeros(n,dtype=int); m=np.random.random(n)<t
            b[m]=np.random.randint(1,8,m.sum()); return b
        l30=atr(N,.08); l60=atr(N,.04); l90=atr(N,.05)
        l30[default==1]+=np.random.randint(0,4,default.sum()).clip(0,10)
        l60[default==1]+=np.random.randint(0,3,default.sum()).clip(0,10)
        l90[default==1]+=np.random.randint(0,4,default.sum()).clip(0,10)
        l30=l30.clip(0,10); l60=l60.clip(0,10); l90=l90.clip(0,10)
        total=l30+l60+l90
        n_open=np.random.poisson(8,N).clip(0,25)
        n_re=np.random.poisson(1.1,N).clip(0,8)
        n_dep=np.random.choice([0,1,2,3,4],N,p=[0.49,.25,.16,.07,.03])
        def perfil(x):
            if x==0: return 'Sem atrasos'
            elif x<=2: return 'Poucos atrasos'
            else: return 'Muitos atrasos'
        df = pd.DataFrame({
            'SeriousDlqin2yrs':default,'age':age,'MonthlyIncome':income,
            'RevolvingUtilizationOfUnsecuredLines':revolving,'DebtRatio':debt,
            'NumberOfTime30-59DaysPastDueNotWorse':l30,
            'NumberOfTime60-89DaysPastDueNotWorse':l60,
            'NumberOfTimes90DaysLate':l90,
            'NumberOfOpenCreditLinesAndLoans':n_open,
            'NumberRealEstateLoansOrLines':n_re,
            'NumberOfDependents':n_dep,'total_atrasos':total,
        })
        df['faixa_etaria'] = pd.cut(df['age'],bins=[17,30,45,60,101],
            labels=['Jovem (18-30)','Adulto (31-45)','Maduro (46-60)','Sênior (61+)'])
        df['faixa_renda'] = pd.cut(df['MonthlyIncome'],bins=[-1,2000,5000,10000,99999999],
            labels=['Baixa (<2k)','Média (2k-5k)','Alta (5k-10k)','Muito Alta (>10k)'])
        df['perfil_historico'] = df['total_atrasos'].apply(perfil)
        return df


@st.cache_resource(show_spinner="Treinando modelos de classificação...")
def treinar_modelos(df_hash):
    df = carregar_dados()
    FEAT = ['age','MonthlyIncome','RevolvingUtilizationOfUnsecuredLines',
            'DebtRatio','NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse','NumberOfTimes90DaysLate',
            'NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines',
            'NumberOfDependents','total_atrasos']
    FEAT = [f for f in FEAT if f in df.columns]
    X = df[FEAT]; y = df['SeriousDlqin2yrs']
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr); X_te_sc = scaler.transform(X_te)

    dt = DecisionTreeClassifier(max_depth=6,min_samples_leaf=50,
                                 class_weight='balanced',random_state=42)
    dt.fit(X_tr, y_tr)

    knn = KNeighborsClassifier(n_neighbors=11, weights='distance', n_jobs=-1)
    knn.fit(X_tr_sc, y_tr)

    nb = GaussianNB(); nb.fit(X_tr, y_tr)

    def met(y_t, y_p, y_prob):
        return dict(
            acc=accuracy_score(y_t,y_p),
            prec=precision_score(y_t,y_p,zero_division=0),
            rec=recall_score(y_t,y_p,zero_division=0),
            f1=f1_score(y_t,y_p,zero_division=0),
            auc=roc_auc_score(y_t,y_prob),
            cm=confusion_matrix(y_t,y_p),
            fpr_tpr=roc_curve(y_t,y_prob)
        )

    y_dt=dt.predict(X_te);   y_dt_p=dt.predict_proba(X_te)[:,1]
    y_knn=knn.predict(X_te_sc); y_knn_p=knn.predict_proba(X_te_sc)[:,1]
    y_nb=nb.predict(X_te);   y_nb_p=nb.predict_proba(X_te)[:,1]

    return dict(
        dt=dt, knn=knn, nb=nb, scaler=scaler, feat=FEAT,
        met_dt=met(y_te,y_dt,y_dt_p),
        met_knn=met(y_te,y_knn,y_knn_p),
        met_nb=met(y_te,y_nb,y_nb_p),
    )


# ─── CARREGAR ─────────────────────────────────────────────────────────────────
df = carregar_dados()
modelos = treinar_modelos(len(df))

AZUL='#2E75B6'; VERM='#E24B4A'; VERDE='#1D9E75'; AMBER='#BA7517'; CINZA='#888888'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-card-front-side.png", width=72)
    st.markdown("### 💳 Risco de Crédito")
    st.markdown("---")
    st.markdown("**Dataset:** Give Me Some Credit")
    st.markdown(f"**Registros:** {len(df):,}")
    inad = df['SeriousDlqin2yrs'].sum()
    st.markdown(f"**Inadimplentes:** {inad:,} ({inad/len(df)*100:.1f}%)")
    st.markdown("---")
    st.markdown("**Modelos treinados:**")
    st.markdown("🌳 Árvore de Decisão")
    st.markdown("🔵 KNN (k=11)")
    st.markdown("📊 Naive Bayes")
    st.markdown("---")
    st.caption("Disciplina: Probabilidade e Classificação")


# ─── TÍTULO ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">💳 Análise de Risco de Crédito</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Dataset: Give Me Some Credit (Kaggle) &nbsp;|&nbsp; Probabilidade e Classificação de Dados</p>', unsafe_allow_html=True)

# KPIs rápidos
k1,k2,k3,k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Total de Registros</div><div class="metric-value">{len(df):,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Taxa de Inadimplência</div><div class="metric-value">{df["SeriousDlqin2yrs"].mean()*100:.1f}%</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">F1 — Árvore</div><div class="metric-value">{modelos["met_dt"]["f1"]:.3f}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">F1 — KNN</div><div class="metric-value">{modelos["met_knn"]["f1"]:.3f}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ─── ABAS PRINCIPAIS ──────────────────────────────────────────────────────────
aba1, aba2 = st.tabs(["📊 Seção 1 — Análise dos Dados", "🤖 Seção 2 — Classificação Interativa"])


# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with aba1:
    st.markdown('<div class="section-header">📈 Exploração e Padrões nos Dados</div>', unsafe_allow_html=True)

    # --- Gráfico 1: Variável alvo ---
    col_a, col_b = st.columns([1,2])
    with col_a:
        st.markdown("**Distribuição da Variável Alvo**")
        st.caption("Objetivo: verificar o desequilíbrio de classes")
        fig, ax = plt.subplots(figsize=(4,3.5))
        cnt = df['SeriousDlqin2yrs'].value_counts().sort_index()
        pct = df['SeriousDlqin2yrs'].value_counts(normalize=True).sort_index()*100
        bars = ax.bar(['Adimplente','Inadimplente'], cnt.values, color=[AZUL,VERM], edgecolor='white', width=0.55)
        for bar,v,p in zip(bars,cnt.values,pct.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
                    f'{v:,}\n({p:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylim(0, max(cnt.values)*1.25)
        ax.set_ylabel('Registros'); ax.set_title('Variável Alvo', fontsize=11, fontweight='bold')
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_b:
        st.markdown("**Taxa de Inadimplência por Faixa Etária**")
        st.caption("Objetivo: identificar grupos etários de maior risco")
        fig, axes = plt.subplots(1,2,figsize=(9,3.5))
        taxa_e = df.groupby('faixa_etaria', observed=True)['SeriousDlqin2yrs'].mean()*100
        cores_e = [VERM if v>8 else AMBER if v>6 else AZUL for v in taxa_e.values]
        axes[0].bar(taxa_e.index, taxa_e.values, color=cores_e, edgecolor='white', width=0.6)
        axes[0].axhline(df['SeriousDlqin2yrs'].mean()*100, color=CINZA, linestyle='--', linewidth=1.2, label='Média')
        for i,(lab,val) in enumerate(zip(taxa_e.index,taxa_e.values)):
            axes[0].text(i, val+0.15, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[0].set_ylim(0, taxa_e.max()*1.3); axes[0].set_ylabel('%'); axes[0].legend(fontsize=8)
        axes[0].set_title('Por Faixa Etária', fontsize=10, fontweight='bold')

        taxa_r = df.groupby('faixa_renda', observed=True)['SeriousDlqin2yrs'].mean()*100
        cores_r = [VERM if v>8 else AMBER if v>6 else VERDE for v in taxa_r.values]
        axes[1].bar(taxa_r.index, taxa_r.values, color=cores_r, edgecolor='white', width=0.6)
        axes[1].axhline(df['SeriousDlqin2yrs'].mean()*100, color=CINZA, linestyle='--', linewidth=1.2)
        for i,(lab,val) in enumerate(zip(taxa_r.index,taxa_r.values)):
            axes[1].text(i, val+0.15, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[1].set_ylim(0, taxa_r.max()*1.3); axes[1].set_ylabel('%')
        axes[1].set_title('Por Faixa de Renda', fontsize=10, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Clientes jovens (18-30) apresentam taxa de inadimplência acima da média geral. Clientes com renda baixa (&lt;$2k/mês) também concentram maior risco. Estas variáveis foram discretizadas para uso no Teorema de Bayes.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # --- Gráfico 2: Histórico de atrasos ---
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Taxa de Inadimplência por Perfil Histórico**")
        st.caption("Objetivo: confirmar o histórico de atrasos como preditor-chave")
        fig, ax = plt.subplots(figsize=(5.5,3.8))
        ordem = ['Sem atrasos','Poucos atrasos','Muitos atrasos']
        tp = df.groupby('perfil_historico')['SeriousDlqin2yrs'].mean().reindex(ordem)*100
        cores_p = [VERDE, AMBER, VERM]
        bars = ax.bar(tp.index, tp.values, color=cores_p, edgecolor='white', width=0.55)
        for bar,val in zip(bars,tp.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylim(0, tp.max()*1.25); ax.set_ylabel('Taxa de inadimplência (%)')
        ax.set_title('Perfil Histórico de Atrasos', fontsize=11, fontweight='bold')
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_d:
        st.markdown("**Mapa de Correlação entre Variáveis**")
        st.caption("Objetivo: identificar variáveis mais relacionadas com inadimplência")
        cols_c2 = ['SeriousDlqin2yrs','age','MonthlyIncome',
                   'RevolvingUtilizationOfUnsecuredLines','total_atrasos','NumberOfDependents']
        cols_c2 = [c for c in cols_c2 if c in df.columns]
        labs_c2 = {'SeriousDlqin2yrs':'Inadimplência','age':'Idade',
                   'MonthlyIncome':'Renda','RevolvingUtilizationOfUnsecuredLines':'Util.Crédito',
                   'total_atrasos':'Total Atrasos','NumberOfDependents':'Dependentes'}
        corr = df[cols_c2].corr()
        corr.index = [labs_c2.get(c,c) for c in corr.index]
        corr.columns = [labs_c2.get(c,c) for c in corr.columns]
        fig, ax = plt.subplots(figsize=(5.5,4.2))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(230,20,as_cmap=True),
                    vmax=0.6, vmin=-0.6, center=0, annot=True, fmt='.2f',
                    annot_kws={'size':9}, square=True, linewidths=0.5, ax=ax,
                    cbar_kws={'shrink':0.8})
        ax.set_title('Correlação de Pearson', fontsize=10, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> <b>total_atrasos</b> e <b>utilização do crédito rotativo</b> são as variáveis com maior correlação com inadimplência — confirmado por EDA, Teorema de Bayes e importância da Árvore de Decisão.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # --- Gráfico 3: Distribuições ---
    st.markdown("**Distribuição de Variáveis-Chave por Classe**")
    st.caption("Objetivo: comparar perfil entre adimplentes e inadimplentes")
    fig, axes = plt.subplots(1, 3, figsize=(13,4))
    vars_dist = [
        ('age','Idade','anos'),
        ('MonthlyIncome','Renda Mensal','USD'),
        ('RevolvingUtilizationOfUnsecuredLines','Util. Crédito','taxa'),
    ]
    for ax, (col, lab, unit) in zip(axes, vars_dist):
        if col not in df.columns: continue
        for cls, cor, lbl in [(0,AZUL,'Adimplente'),(1,VERM,'Inadimplente')]:
            sub = df[df['SeriousDlqin2yrs']==cls][col]
            ax.hist(sub.clip(sub.quantile(.01), sub.quantile(.99)),
                    bins=35, color=cor, alpha=0.6, label=lbl, edgecolor='white', density=True)
        ax.set_xlabel(f'{lab} ({unit})'); ax.set_ylabel('Densidade')
        ax.set_title(lab, fontsize=11, fontweight='bold'); ax.legend(fontsize=9)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # --- Gráfico 4: Desempenho dos modelos ---
    st.markdown("---")
    st.markdown('<div class="section-header">🏆 Desempenho dos Modelos de Classificação</div>', unsafe_allow_html=True)
    col_e, col_f = st.columns([3,2])

    with col_e:
        st.markdown("**Comparação de Métricas**")
        st.caption("Objetivo: identificar qual modelo tem melhor desempenho por critério")
        met_nomes = ['Acurácia','Precisão','Recall','F1-Score','AUC-ROC']
        v_dt  = [modelos['met_dt']['acc'],  modelos['met_dt']['prec'],  modelos['met_dt']['rec'],  modelos['met_dt']['f1'],  modelos['met_dt']['auc']]
        v_knn = [modelos['met_knn']['acc'], modelos['met_knn']['prec'], modelos['met_knn']['rec'], modelos['met_knn']['f1'], modelos['met_knn']['auc']]
        v_nb  = [modelos['met_nb']['acc'],  modelos['met_nb']['prec'],  modelos['met_nb']['rec'],  modelos['met_nb']['f1'],  modelos['met_nb']['auc']]
        fig, ax = plt.subplots(figsize=(8,4))
        x = np.arange(len(met_nomes)); w = 0.25
        for bars, vals, cor, lbl in [
            (ax.bar(x-w, v_dt, w, color=AZUL, edgecolor='white'), v_dt, AZUL, 'Árvore de Decisão'),
            (ax.bar(x,  v_knn, w, color=VERM, edgecolor='white'), v_knn, VERM, 'KNN (k=11)'),
            (ax.bar(x+w, v_nb, w, color=VERDE, edgecolor='white'), v_nb, VERDE, 'Naive Bayes'),
        ]:
            bars.set_label(lbl)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.006,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(met_nomes, fontsize=10)
        ax.set_ylim(0, 1.15); ax.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with col_f:
        st.markdown("**Curvas ROC**")
        st.caption("Objetivo: avaliar capacidade discriminante independente do limiar")
        fig, ax = plt.subplots(figsize=(5,4))
        for met_k, cor, nome, ls in [
            ('met_dt', AZUL, 'Árvore', '-'),
            ('met_knn', VERM, 'KNN', '-'),
            ('met_nb', VERDE, 'Naive Bayes', '--'),
        ]:
            fpr, tpr, _ = modelos[met_k]['fpr_tpr']
            ax.plot(fpr, tpr, color=cor, lw=2, ls=ls,
                    label=f"{nome} ({modelos[met_k]['auc']:.3f})")
        ax.plot([0,1],[0,1], color=CINZA, ls=':', lw=1.2, label='Aleatório')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR (Recall)')
        ax.legend(fontsize=9, loc='lower right')
        ax.set_title('Curvas ROC', fontsize=11, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — CLASSIFICAÇÃO INTERATIVA
# ══════════════════════════════════════════════════════════════════════════════
with aba2:
    st.markdown('<div class="section-header">🎯 Classifique um Novo Cliente</div>', unsafe_allow_html=True)
    st.markdown("Preencha os atributos do cliente abaixo e clique em **Classificar** para obter a predição dos três métodos.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Dados Pessoais**")
        idade = st.slider("Idade", 18, 95, 35)
        dependentes = st.slider("Número de dependentes", 0, 10, 0)
        faixa_et = st.selectbox("Faixa etária", ['Jovem (18-30)','Adulto (31-45)','Maduro (46-60)','Sênior (61+)'],
                                index=1)

    with col2:
        st.markdown("**💰 Situação Financeira**")
        renda = st.number_input("Renda mensal (USD)", min_value=0, max_value=50000, value=5000, step=100)
        faixa_rd = st.selectbox("Faixa de renda",
                                ['Baixa (<2k)','Média (2k-5k)','Alta (5k-10k)','Muito Alta (>10k)'],
                                index=1)
        debt_ratio = st.slider("Debt Ratio (dívida/renda)", 0.0, 3.0, 0.3, step=0.05)
        util_credito = st.slider("Utilização do crédito rotativo (0-1)", 0.0, 1.5, 0.2, step=0.01)

    with col3:
        st.markdown("**📋 Histórico de Crédito**")
        atr_30 = st.slider("Atrasos 30-59 dias", 0, 10, 0)
        atr_60 = st.slider("Atrasos 60-89 dias", 0, 10, 0)
        atr_90 = st.slider("Atrasos ≥ 90 dias", 0, 10, 0)
        linhas  = st.slider("Linhas de crédito abertas", 0, 25, 8)
        imov    = st.slider("Empréstimos imobiliários", 0, 10, 1)

    total_at = atr_30 + atr_60 + atr_90
    perf_hist = 'Sem atrasos' if total_at==0 else ('Poucos atrasos' if total_at<=2 else 'Muitos atrasos')

    st.info(f"📊 **Perfil calculado:** Total de atrasos = {total_at} → **{perf_hist}**")

    st.markdown("---")
    btn = st.button("🔍 Classificar Cliente", type="primary", use_container_width=True)

    if btn:
        # ── TEOREMA DE BAYES (manual) ────────────────────────────────────────
        p_inad = df['SeriousDlqin2yrs'].mean()
        p_adim = 1 - p_inad

        def veros(col, cat, classe):
            sub = df[df['SeriousDlqin2yrs']==classe]
            freq = sub[col].value_counts(normalize=True)
            return float(freq.get(cat, 1e-6))

        num1 = p_inad * veros('faixa_etaria',faixa_et,1) * veros('faixa_renda',faixa_rd,1) * veros('perfil_historico',perf_hist,1)
        num0 = p_adim * veros('faixa_etaria',faixa_et,0) * veros('faixa_renda',faixa_rd,0) * veros('perfil_historico',perf_hist,0)
        soma = num1 + num0
        prob_bayes_inad = num1 / soma if soma > 0 else 0
        prob_bayes_adim = num0 / soma if soma > 0 else 1

        # ── CLASSIFICADORES ──────────────────────────────────────────────────
        X_novo = pd.DataFrame([{
            'age': idade,
            'MonthlyIncome': renda,
            'RevolvingUtilizationOfUnsecuredLines': util_credito,
            'DebtRatio': debt_ratio,
            'NumberOfTime30-59DaysPastDueNotWorse': atr_30,
            'NumberOfTime60-89DaysPastDueNotWorse': atr_60,
            'NumberOfTimes90DaysLate': atr_90,
            'NumberOfOpenCreditLinesAndLoans': linhas,
            'NumberRealEstateLoansOrLines': imov,
            'NumberOfDependents': dependentes,
            'total_atrasos': total_at,
        }])
        X_novo = X_novo[[f for f in modelos['feat'] if f in X_novo.columns]]
        X_novo_sc = modelos['scaler'].transform(X_novo)

        prob_dt_inad  = modelos['dt'].predict_proba(X_novo)[0,1]
        prob_knn_inad = modelos['knn'].predict_proba(X_novo_sc)[0,1]

        pred_dt  = 1 if prob_dt_inad  >= 0.5 else 0
        pred_knn = 1 if prob_knn_inad >= 0.5 else 0
        pred_bay = 1 if prob_bayes_inad >= 0.5 else 0

        # ── RESULTADOS ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Resultado da Classificação")

        r1, r2, r3 = st.columns(3)

        def render_pred(col, nome, emoji, prob_inad, pred, cor_nome):
            with col:
                st.markdown(f"**{emoji} {nome}**")
                cls = "pred-inadim" if pred==1 else "pred-adim"
                txt = "⚠️ INADIMPLENTE" if pred==1 else "✅ ADIMPLENTE"
                st.markdown(f'<div class="pred-box {cls}">{txt}</div>', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(3.5,2.2))
                prob_adim = 1 - prob_inad
                ax2.barh(['Adimplente','Inadimplente'], [prob_adim*100, prob_inad*100],
                         color=[AZUL, VERM], edgecolor='white', height=0.5)
                for i, val in enumerate([prob_adim*100, prob_inad*100]):
                    ax2.text(val+0.5, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
                ax2.set_xlim(0,115); ax2.axvline(50, color=CINZA, ls='--', lw=1)
                ax2.set_title('Probabilidade', fontsize=10, fontweight='bold')
                st.pyplot(fig2, use_container_width=True); plt.close()

        render_pred(r1, "Teorema de Bayes", "🧮", prob_bayes_inad, pred_bay, VERDE)
        render_pred(r2, "Árvore de Decisão", "🌳", prob_dt_inad,  pred_dt,  AZUL)
        render_pred(r3, "KNN (k=11)",        "🔵", prob_knn_inad, pred_knn, VERM)

        # ── COMPARAÇÃO VISUAL ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔄 Comparação Visual dos Três Métodos")
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        metodos = ['Bayes\n(Manual)', 'Árvore de\nDecisão', 'KNN\n(k=11)']
        probs_i = [prob_bayes_inad*100, prob_dt_inad*100, prob_knn_inad*100]
        probs_a = [100-v for v in probs_i]
        bars_a = ax3.bar(metodos, probs_a, color=AZUL, edgecolor='white', width=0.5, label='Adimplente')
        bars_i = ax3.bar(metodos, probs_i, bottom=probs_a, color=VERM, edgecolor='white', width=0.5, label='Inadimplente')
        for i, (pa, pi) in enumerate(zip(probs_a, probs_i)):
            if pi > 4:
                ax3.text(i, pa + pi/2, f'{pi:.1f}%', ha='center', va='center',
                         fontsize=12, fontweight='bold', color='white')
            if pa > 4:
                ax3.text(i, pa/2, f'{pa:.1f}%', ha='center', va='center',
                         fontsize=12, fontweight='bold', color='white')
        ax3.axhline(50, color='white', ls='--', lw=2, alpha=0.7)
        ax3.set_ylabel('Probabilidade (%)'); ax3.set_ylim(0, 115)
        ax3.set_title('Probabilidade de Inadimplência — Comparação dos Três Métodos',
                      fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10, loc='upper right')
        plt.tight_layout(); st.pyplot(fig3, use_container_width=True); plt.close()

        # ── CONSENSO ─────────────────────────────────────────────────────────
        votos_inad = pred_bay + pred_dt + pred_knn
        if votos_inad >= 2:
            st.error(f"⚠️ **Consenso: ALTO RISCO DE INADIMPLÊNCIA** — {votos_inad}/3 modelos classificam como inadimplente.")
        else:
            st.success(f"✅ **Consenso: BAIXO RISCO** — {3-votos_inad}/3 modelos classificam como adimplente.")

        # ── DETALHES DO BAYES ─────────────────────────────────────────────────
        with st.expander("🧮 Ver cálculo detalhado do Teorema de Bayes"):
            p_et_1 = veros('faixa_etaria',  faixa_et,  1)
            p_et_0 = veros('faixa_etaria',  faixa_et,  0)
            p_rd_1 = veros('faixa_renda',   faixa_rd,  1)
            p_rd_0 = veros('faixa_renda',   faixa_rd,  0)
            p_ph_1 = veros('perfil_historico', perf_hist, 1)
            p_ph_0 = veros('perfil_historico', perf_hist, 0)
            st.markdown(f"""
**Fórmula:** P(C | X) = P(C) × P(X₁|C) × P(X₂|C) × P(X₃|C) / Σ[...]

**Variáveis usadas:**
- Faixa etária: `{faixa_et}`
- Faixa de renda: `{faixa_rd}`
- Perfil histórico: `{perf_hist}`

**Cálculo para C=1 (Inadimplente):**
```
P(C=1) = {p_inad:.4f}
× P({faixa_et} | C=1) = {p_et_1:.4f}
× P({faixa_rd} | C=1) = {p_rd_1:.4f}
× P({perf_hist} | C=1) = {p_ph_1:.4f}
────────────────────────────────
Numerador C=1 = {num1:.8f}
```

**Cálculo para C=0 (Adimplente):**
```
P(C=0) = {p_adim:.4f}
× P({faixa_et} | C=0) = {p_et_0:.4f}
× P({faixa_rd} | C=0) = {p_rd_0:.4f}
× P({perf_hist} | C=0) = {p_ph_0:.4f}
────────────────────────────────
Numerador C=0 = {num0:.8f}
```

**Probabilidade posterior:**
```
Soma (evidência) = {soma:.8f}
P(inadimplente | X) = {num1:.8f} / {soma:.8f} = {prob_bayes_inad*100:.2f}%
P(adimplente   | X) = {num0:.8f} / {soma:.8f} = {prob_bayes_adim*100:.2f}%
```
""")
