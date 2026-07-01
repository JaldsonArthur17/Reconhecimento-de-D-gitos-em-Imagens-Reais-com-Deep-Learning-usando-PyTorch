import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard - Projeto SVHN", layout="wide")

dados_evolucao = [
    {"Fase": "1. Baseline CNN", "Acuracia": 87.88, "Loss": 0.29, "Imagens": 73257, "Tecnica": "Supervisionado Simples"},
    {"Fase": "2. Baseline + Batch Norm", "Acuracia": 90.35, "Loss": 0.29, "Imagens": 73257, "Tecnica": "Estabilização Matemática"},
    {"Fase": "3. DeepCNN", "Acuracia": 94.23, "Loss": 0.18, "Imagens": 73257, "Tecnica": "Rede Profunda + Dropout"},
    {"Fase": "4. RandAugment", "Acuracia": 95.30, "Loss": 0.21, "Imagens": 73257, "Tecnica": "Data Augmentation Agressivo"},
    {"Fase": "5. FixMatch (50k Extras)", "Acuracia": 94.63, "Loss": 0.05, "Imagens": 123257, "Tecnica": "Semi-Supervisionado"},
    {"Fase": "6. FixMatch (100k Extras)", "Acuracia": 95.16, "Loss": 0.03, "Imagens": 173257, "Tecnica": "Semi-Supervisionado Escalonado"},
    {"Fase": "7. FixMatch (200k Extras)", "Acuracia": 95.50, "Loss": 0.03, "Imagens": 273257, "Tecnica": "Ajuste Fino Profundo"},
    
    {"Fase": "8. FixMatch (300k Extras)", "Acuracia": 95.83, "Loss": 0.03, "Imagens": 373257, "Tecnica": "Escala Massiva de Dados (Pico)"},
    {"Fase": "9. FixMatch (400k Extras)", "Acuracia": 95.76, "Loss": 0.03, "Imagens": 473257, "Tecnica": "Limite Computacional (Saturação)"},
    
]

df = pd.DataFrame(dados_evolucao)

st.title("Evolução Arquitetural - Deep Learning no Dataset SVHN")
st.markdown("""
Confira o avanço do nosso modelo de visão computacional. 
Observe como a **Acurácia** aumenta e o **Erro (Loss)** diminui à medida que passamos de uma rede simples para abordagens avançadas de aprendizado Semi-Supervisionado (FixMatch) utilizando imagens não rotuladas.
""")
st.divider()

col1, col2, col3 = st.columns(3)

fase_final = df.loc[df["Acuracia"].idxmax()]

col1.metric("Acurácia Máxima Atingida", f"{fase_final['Acuracia']}%", f"+{fase_final['Acuracia'] - df.iloc[0]['Acuracia']:.2f}% desde a Baseline")
col2.metric("Redução do Erro (Loss)", f"{fase_final['Loss']:.2f}", f"{fase_final['Loss'] - df.iloc[0]['Loss']:.2f} comparado ao início", delta_color="inverse")
col3.metric("Volume de Dados (Treino)", f"{fase_final['Imagens']:,}".replace(',', '.'), "Imagens processadas")

st.divider()

st.subheader("Gráfico de Evolução")

tab1, tab2 = st.tabs(["Acurácia vs Fases do Projeto", "Impacto do Volume de Dados"])

with tab1:
    fig_acc = go.Figure()
    
    fig_acc.add_trace(go.Scatter(x=df['Fase'], y=df['Acuracia'], mode='lines+markers+text',
                                 name='Acurácia (%)', line=dict(color='#00b4d8', width=3),
                                 text=df['Acuracia'].apply(lambda x: f"{x}%"), textposition="top center"))
    
    fig_acc.add_trace(go.Scatter(x=df['Fase'], y=df['Loss'], mode='lines+markers',
                                 name='Loss', yaxis='y2', line=dict(color='#ff4d4d', width=3, dash='dot')))

    fig_acc.update_layout(
        title='Crescimento da Acurácia e Redução de Erro por Etapa',
        xaxis_title='Etapas do Projeto',
        yaxis_title='Acurácia (%)',
        yaxis2=dict(title='Loss (Erro)', overlaying='y', side='right'),
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig_acc, use_container_width=True)

with tab2:
    fig_dados = px.scatter(df, x="Imagens", y="Acuracia", size="Acuracia", color="Tecnica",
                           hover_name="Fase", text="Fase", size_max=20,
                           title="Correlação entre Volume de Imagens e Acurácia (O Paradoxo de FixMatch)")
    fig_dados.update_traces(textposition='bottom center')
    fig_dados.update_layout(height=500)
    st.plotly_chart(fig_dados, use_container_width=True)

st.divider()
st.subheader("Tabela de Registros do Treinamento")
st.dataframe(df.style.highlight_max(subset=['Acuracia'], color='lightgreen')
                    .highlight_min(subset=['Loss'], color='lightgreen'), 
             use_container_width=True)

st.markdown("""
### Conclusão Científica:
O experimento comprova que redes neurais convolucionais (CNNs) escalam o seu poder de generalização não apenas através do aumento de parâmetros matemáticos (DeepCNN), mas **crucialmente através da injeção de dados do mundo real**. 
O uso do **FixMatch** provou que o modelo é capaz de "ensinar a si mesmo" utilizando imagens sem rótulo, quebrando a barreira dos 95% de precisão de forma semi-supervisionada.
""")