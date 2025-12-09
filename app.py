import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from math import radians, cos, sin, asin, sqrt

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Sonda Delivery ML", page_icon="🚚", layout="wide")

# --- 1. CARREGAMENTO DOS ARQUIVOS ---
@st.cache_resource
def load_assets():
    try:
        # Carrega o modelo e a base de CEPs
        model = joblib.load('modelo_entregas.joblib')
        geo_data = pd.read_csv('referencia_geo.csv')
        geo_data['geolocation_zip_code_prefix'] = geo_data['geolocation_zip_code_prefix'].astype(str).str.zfill(5)
        
        # Tenta carregar o arquivo de comparação para o gráfico (pode não existir na primeira execução)
        try:
            df_comp = pd.read_csv('comparativo_modelo.csv')
        except:
            df_comp = None
            
        return model, geo_data, df_comp
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None, None

model, geo_data, df_comp = load_assets()

# --- 2. FUNÇÕES AUXILIARES ---
def get_lat_lon(cep, geo_df):
    # Formata o CEP para pegar os 5 primeiros dígitos
    prefixo = str(cep).replace("-", "").replace(".", "").strip()[:5]
    row = geo_df[geo_df['geolocation_zip_code_prefix'] == prefixo]
    if not row.empty:
        return row.iloc[0]['geolocation_lat'], row.iloc[0]['geolocation_lng']
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Raio da terra em km
    phi1, phi2 = map(radians, [lat1, lat2])
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2*R*asin(sqrt(a))

# --- 3. INTERFACE VISUAL ---
st.title("🚚 Sonda Delivery: Otimização Logística com ML")
st.markdown("---")

# Ordem das Abas: Performance Primeiro, Simulador Depois (como você configurou)
tab1, tab2, tab3 = st.tabs(["📈 Performance do Modelo", "🧮 Simulador de Prazo", "🚀 Impacto & Futuro"])

# ==============================================================================
# ABA 1: PERFORMANCE (Gráfico de Comparação e Importância)
# ==============================================================================
with tab1:
    st.header("Análise de Performance e Explicabilidade")

    # --- 1. GRÁFICO DE ERRO (Agora vem PRIMEIRO) ---
    st.subheader("📉 Distribuição dos Erros (Legado vs ML)")
    st.caption("Comparativo de quantos dias cada sistema erra. O ideal é que o gráfico esteja alto e centralizado no 0.")
    
    if df_comp is not None:
        # Calcular Erros
        df_comp['Erro Antigo'] = df_comp['dias_reais'] - df_comp['dias_estimados_antigo']
        df_comp['Erro ML'] = df_comp['dias_reais'] - df_comp['dias_previstos_ia']
        
        # KPIs
        mae_antigo = df_comp['Erro Antigo'].abs().mean()
        mae_novo = df_comp['Erro ML'].abs().mean()
        melhoria = ((mae_antigo - mae_novo) / mae_antigo) * 100
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Erro Médio (Antigo)", f"{mae_antigo:.1f} dias", delta_color="off")
        k2.metric("Erro Médio (ML)", f"{mae_novo:.1f} dias", delta=f"{melhoria:.1f}% melhor", delta_color="normal")
        k3.metric("Amostra Analisada", f"{len(df_comp)} pedidos")
        
        # Histograma
        df_long = pd.melt(df_comp[['Erro Antigo', 'Erro ML']], var_name='Modelo', value_name='Dias de Erro')
        fig_hist = px.histogram(df_long, x="Dias de Erro", color="Modelo",
                           nbins=100, range_x=[-20, 20], opacity=0.6, barmode="overlay",
                           color_discrete_map={'Erro Antigo': '#FF4B4B', 'Erro ML': '#00CC96'})
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    else:
        st.warning("⚠️ Arquivo 'comparativo_modelo.csv' não encontrado.")

    st.divider()

    # --- 2. GRÁFICO DE IMPORTÂNCIA (Agora vem DEPOIS) ---
    st.subheader("🧠 Por que o ML tomou essa decisão?")
    st.markdown("O gráfico abaixo mostra quais variáveis têm maior peso no cálculo do prazo.")

    if model is not None:
        # Extrair a importância das features do modelo treinado
        importancias = model.feature_importances_
        # Nomes das colunas usadas no treino (nessa ordem exata)
        features = ['Distância (km)', 'Peso (g)', 'Volume (cm³)', 'Valor do Frete (R$)', 'Preço do Produto (R$)']
        
        # Criar DataFrame para o gráfico
        df_imp = pd.DataFrame({'Fator': features, 'Importância (%)': importancias * 100})
        df_imp = df_imp.sort_values('Importância (%)', ascending=True) # Ordenar para o gráfico
        
        # Gráfico de Barras Horizontais
        fig_imp = px.bar(df_imp, x='Importância (%)', y='Fator', orientation='h',
                         text_auto='.1f', # Mostra o valor na barra
                         color='Importância (%)', 
                         color_continuous_scale='Blues')
        
        fig_imp.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.info("""
        **Interpretação:**
        * **Distância:** Geralmente é o fator #1 (Física).
        * **Valor do Frete:** O ML aprende que frete mais caro geralmente indica modal expresso (Sedex/Transportadora Rápida), reduzindo o prazo.
        """)

# ==============================================================================
# ABA 2: SIMULADOR (Operacional)
# ==============================================================================
with tab2:
    st.subheader("Simulação de Entrega em Tempo Real")
    st.markdown("Preencha os dados da rota para estimar o prazo com o ML.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📍 **Rota**")
        cep_origem = st.text_input("CEP Origem (Vendedor)", "13023") # Ex: Campinas
        cep_destino = st.text_input("CEP Destino (Cliente)", "42800") # Ex: Bahia
        
    with col2:
        st.info("📦 **Pacote**")
        peso = st.number_input("Peso (gramas)", value=225)
        # Volume aproximado em cm3
        volume = st.number_input("Volume (cm³)", value=2000, help="Altura x Largura x Comprimento")
        
    with col3:
        st.info("💰 **Financeiro**")
        frete = st.number_input("Valor do Frete (R$)", value=20.0)
        preco = st.number_input("Preço do Produto (R$)", value=100.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Calcular Prazo Real", use_container_width=True):
        if model and geo_data is not None:
            # 1. Obter Coordenadas
            lat_origem, lon_origem = get_lat_lon(cep_origem, geo_data)
            lat_destino, lon_destino = get_lat_lon(cep_destino, geo_data)
            
            if lat_origem and lat_destino:
                # 2. Calcular Distância
                distancia = haversine(lat_origem, lon_origem, lat_destino, lon_destino)
                
                # 3. Preparar dados para o Modelo (Mesma ordem do treinamento!)
                # Features: ['distancia_km', 'product_weight_g', 'volume_cm3', 'freight_value', 'price']
                dados_entrada = pd.DataFrame([[distancia, peso, volume, frete, preco]], 
                                           columns=['distancia_km', 'product_weight_g', 'volume_cm3', 'freight_value', 'price'])
                
                # 4. Predição
                prazo_estimado = model.predict(dados_entrada)[0]
                
                # 5. Exibir Resultado
                st.success("Cálculo realizado com sucesso!")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Distância Aérea", f"{distancia:.1f} km")
                m2.metric("Prazo Estimado (ML)", f"{prazo_estimado:.1f} dias")
                m3.metric("Prazo Conservador (Legacy)", f"{prazo_estimado + 7:.0f} dias", delta="-7 dias", delta_color="inverse")
                
            else:
                st.error("CEP não encontrado na base de dados.")
        else:
            st.error("Modelo não carregado. Verifique os arquivos .joblib e .csv")
    
# ==============================================================================
# ABA 3: IMPACTO E FUTURO (NOVA!)
# ==============================================================================
with tab3:
    st.header("Visão Estratégica: Próximos Passos")
    st.markdown("O modelo atual é apenas o começo. Abaixo detalhamos o potencial de geração de valor e o roadmap técnico.")

    col_business, col_tech = st.columns(2, gap="large")

    # Função para criar cartões personalizados com texto branco
    def card(icon, title, text, bg_color):
        st.markdown(f"""
        <div style="
            background-color: {bg_color};
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            color: white;  /* FORÇA A COR BRANCA NO TEXTO */
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
            <h4 style="color: white; margin: 0 0 10px 0;">{icon} {title}</h4>
            <p style="color: #f0f0f0; margin: 0; font-size: 16px;">{text}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- COLUNA DE NEGÓCIOS (VERDE ESCURO) ---
    with col_business:
        st.subheader("🎯 Expectativas de Impacto Direto")
        st.caption("Benefícios financeiros e operacionais")
        
        # Cor de fundo: Verde Floresta (#2E7D32) para contraste com texto branco
        bg_biz = "#2E7D32" 
        
        card("✅", "Fim dos 'Colchões'", 
             "Estimativas precisas eliminam a necessidade de adicionar dias extras de segurança. O prazo informado é o prazo real.", bg_biz)
        
        card("🤝", "Aumento da Confiança", 
             "O cliente recebe um prazo realista. Cumprir a promessa exata gera mais fidelidade do que prometer longe.", bg_biz)
        
        card("🛒", "Menos Abandono", 
             "Em regiões próximas, o modelo reduz o prazo informado, convertendo clientes que desistiriam com prazos longos.", bg_biz)
        
        card("🚚", "Competitividade", 
             "Nossa oferta de frete se torna mais atraente frente aos concorrentes, sem aumentar o custo operacional.", bg_biz)

    # --- COLUNA TÉCNICA (AZUL ESCURO) ---
    with col_tech:
        st.subheader("🛠️ Melhorias Técnicas Planejadas")
        st.caption("Roadmap de evolução da IA")
        
        # Cor de fundo: Azul Navy (#1565C0) para contraste com texto branco
        bg_tech = "#1565C0"
        
        card("🌊", "Sazonalidade", 
             "Inclusão de variáveis temporais (Black Friday, Natal) para prever gargalos em datas críticas.", bg_tech)
        
        card("🗺️", "Granularidade por CEP", 
             "Uso de dados de volume por região para identificar áreas de risco recorrente.", bg_tech)
        
        card("🤖", "Modelos Robustos", 
             "Teste de algoritmos como XGBoost e integração de dados de trânsito em tempo real.", bg_tech)
        
        card("🔄", "MLOps (Monitoramento)", 
             "Retreino automático mensal para adaptação a mudanças na malha logística.", bg_tech)

    st.divider()
    st.markdown("**Conclusão:** A implementação deste modelo é uma mudança de paradigma na experiência de compra.")