
import streamlit as st
import joblib
from google import genai
from google.genai import types
import os

# Configuração inicial da página web
st.set_page_config(
    page_title="Assistente Editorial e Revisor Ético",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Assistente Editorial & Revisor Ético para Jornalismo")
st.markdown("Esta aplicação auxilia redatores e editores a produzirem matérias mais neutras, profissionais e alinhadas com as diretrizes de ética jornalística do **Manual Universa**.")

# Função para carregar o modelo de Machine Learning local com cache
@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load('modelo_jornalismo.pkl')
    except:
        return None

modelo_local = carregar_modelo()

# Barra lateral - Configuração de Credenciais da API
st.sidebar.header("⚙️ Configuração de Credenciais")
api_key_input = st.sidebar.text_input("Insira sua Gemini API Key:", type="password")

# Interface em Abas
aba_ia, aba_local = st.tabs(["✨ Revisor Ético Avançado (Gemini)", "📊 Classificador Estatístico Local"])

with aba_ia:
    st.header("Revisor Ético de Matérias (Manual Universa)")
    st.write("Analisa profundamente rascunhos de notícias sobre violência contra a mulher.")
    
    texto_ia = st.text_area(
        "Cole o rascunho da matéria para revisão ética:",
        placeholder="Ex: Uma jovem diz ter sido estuprada ontem à noite...",
        height=200,
        key="input_ia"
    )
    
    if st.button("Iniciar Análise de IA", type="primary"):
        if not api_key_input:
            st.warning("🔑 Por favor, insira sua chave da API do Gemini na barra lateral esquerda para prosseguir.")
        elif not texto_ia.strip():
            st.warning("Digite ou cole uma matéria para analisar.")
        else:
            with st.spinner("Analisando com base no Manual Universa..."):
                try:
                    client = genai.Client(api_key=api_key_input)
                    
                    # Definido como uma linha única contínua usando representação de quebra de linha para evitar quebra de literal física
                    instrucao_sistema = (
                        "Você é o Assistente Editorial e Revisor Ético de Jornalismo, especializado na orientação de jornalistas "
                        "e redatores para a cobertura responsável de crimes e pautas sobre violência contra a mulher, seguindo "
                        "estritamente o Manual Universa de Boas Práticas na Cobertura da Violência contra a Mulher.

"
                        "Sua função é orientar repórteres durante o planejamento, apuração, redação e revisão de rascunhos de matérias.

"
                        "Ao revisar o texto fornecido, exija estritamente o cumprimento das regras:
"
                        "1. **As 5 Regras de Ouro**: Conhecer a legislação, jamais culpabilizar a vítima, não justificar o agressor "
                        "(como alegar ciúmes, bebida ou descontrole), evitar o sensacionalismo/morbidez e amparar-se legalmente.
"
                        "2. **Vocabulário Ético**: Corrija termos inadequados (mude 'crime passional' para 'feminicídio'; "
                        "mude 'mulher diz ter sido estuprada' para 'mulher denuncia estupro'; preserve a nomenclatura jurídica).
"
                        "3. **Canais de Apoio**: Exija sempre a indicação de canais de denúncia e acolhimento como o Ligue 180 ou 190.

"
                        "Forneça um feedback bem estruturado ao jornalista apontando os desvios éticos ou de linguagem e sugira a reescrita correta."
                    )
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash-lite',
                        contents=texto_ia,
                        config=types.GenerateContentConfig(
                            system_instruction=instrucao_sistema,
                            temperature=0.2,
                        )
                    )
                    
                    st.subheader("📝 Feedback do Revisor Ético:")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Erro ao processar requisição: {e}")

with aba_local:
    st.header("Classificador Estatístico Offline (ML)")
    st.write("Modelo estatístico rápido de Machine Learning treinado localmente no servidor.")
    
    if modelo_local is None:
        st.error("❌ Arquivo 'modelo_jornalismo.pkl' não foi encontrado no servidor da aplicação.")
    else:
        texto_local = st.text_area(
            "Cole o texto para verificar sensibilidade de termos:",
            placeholder="Ex: O prefeito assinou o decreto de pavimentação das vias...",
            height=150,
            key="input_local"
        )
        
        if st.button("Verificar Sensibilidade"):
            if not texto_local.strip():
                st.warning("Digite algum texto para realizar a verificação.")
            else:
                predicao = modelo_local.predict([texto_local])[0]
                probabilidade = modelo_local.predict_proba([texto_local])[0]
                
                if predicao == 1:
                    st.error(f"⚠️ **ALERTA**: Este texto pode conter termos inadequados, sensacionalistas ou tendenciosos! (Confiança: {probabilidade[1]:.2%})")
                else:
                    st.success(f"✅ **SEGURO**: O texto possui um tom neutro e profissional adequado para a publicação. (Confiança: {probabilidade[0]:.2%})")
