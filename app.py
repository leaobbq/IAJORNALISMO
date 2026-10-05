import streamlit as st
import joblib
import re

# Configuração inicial da página web
st.set_page_config(
    page_title="Revisor Editorial Ético Local",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Revisor Editorial & Classificador Ético Gratuito")
st.markdown("Esta aplicação funciona de forma **100% local e gratuita** (sem uso de APIs pagas ou chaves do Gemini). Ela utiliza o modelo estatístico treinado com as diretrizes do **Manual Universa** para analisar textos jornalísticos.")

# Função para carregar o modelo de Machine Learning local
@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load('modelo_jornalismo.pkl')
    except:
        return None

modelo_local = carregar_modelo()

if modelo_local is None:
    st.error("❌ Arquivo 'modelo_jornalismo.pkl' não foi encontrado. Certifique-se de subir este arquivo no seu GitHub junto com o app.py!")
else:
    # Interface em Abas
    aba_revisor, aba_classificador = st.tabs(["🔍 Revisor Ético de Matérias (Local)", "📊 Classificador Rápido de Sentenças"])

    with aba_revisor:
        st.header("Revisor Ético de Textos Completos")
        st.write("O sistema analisa seu rascunho de matéria dividindo-o período por período para identificar possíveis desvios éticos ou sensacionalistas.")

        texto_materia = st.text_area(
            "Cole o rascunho completo da sua notícia aqui:",
            placeholder="Cole aqui o rascunho da matéria para revisão offline...",
            height=250,
            key="input_revisor"
        )

        if st.button("Analisar Matéria Inteira", type="primary"):
            if not texto_materia.strip():
                st.warning("Por favor, digite ou cole um texto para ser analisado.")
            else:
                # Dividindo o texto por sentenças de forma simples usando pontuação (. ! ?)
                sentencas = [s.strip() for s in re.split(r'[.!?\n]+', texto_materia) if len(s.strip()) > 5]

                st.subheader("📋 Relatório de Análise Ética (Local)")

                trechos_alertas = []
                trechos_seguros = []

                for s in sentencas:
                    predicao = modelo_local.predict([s])[0]
                    probabilidade = modelo_local.predict_proba([s])[0]

                    if predicao == 1:
                        trechos_alertas.append((s, probabilidade[1]))
                    else:
                        trechos_seguros.append((s, probabilidade[0]))

                if trechos_alertas:
                    st.error(f"⚠️ Identificamos {len(trechos_alertas)} trecho(s) com forte indício de desvio ético ou sensacionalismo:")
                    for trecho, prob in trechos_alertas:
                        st.markdown(f"* '{trecho}' — **(Confiança: {prob:.2%})**")
                        st.caption("💡 *Dica do Manual Universa: Evite termos que culpabilizem a vítima, que tentem atenuar o crime justificando o comportamento do agressor (ex: ciúmes, bebida), ou termos antigos como 'crime passional'. Recomenda-se o uso de termos técnicos e objetivos (como 'feminicídio' ou 'agressão'). Lembre-se de adicionar canais de acolhimento como o Ligue 180.*")
                        st.write("---")
                else:
                    st.success("✅ Nenhuma frase com desvios éticos explícitos foi encontrada pelo modelo local! O texto parece seguir um tom profissional e ético.")

                if trechos_seguros:
                    with st.expander("Visualizar trechos identificados como seguros"):
                        for trecho, prob in trechos_seguros:
                            st.write(f"✔️ '{trecho}' — (Seguro com {prob:.2%})")

    with aba_classificador:
        st.header("Classificador Estatístico Rápido")
        st.write("Teste a sensibilidade de uma frase de forma direta e veja a probabilidade calculada pelo modelo estatístico.")

        frase_teste = st.text_input(
            "Digite uma frase ou termo para testar:",
            placeholder="Ex: O homem matou a esposa motivado por ciúmes."
        )

        if st.button("Classificar Frase"):
            if not frase_teste.strip():
                st.warning("Por favor, insira uma frase para classificar.")
            else:
                predicao = modelo_local.predict([frase_teste])[0]
                probabilidade = modelo_local.predict_proba([frase_teste])[0]

                if predicao == 1:
                    st.error(f"⚠️ **ALERTA**: Esta frase pode conter termos inadequados, sensacionalistas ou violadores das diretrizes éticas. (Probabilidade de desvio: {probabilidade[1]:.2%})")
                else:
                    st.success(f"✅ **SEGURO**: Esta frase possui um tom neutro, objetivo e profissional de acordo com o modelo. (Confiança: {probabilidade[0]:.2%})")
