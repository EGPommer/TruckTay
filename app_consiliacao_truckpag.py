import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import re
import io
from datetime import datetime

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Sistema de Conciliação - Degranel / Truckpag",
    page_icon="🚛",
    layout="wide"
)


# ==============================================================================
# MÓDULO A: HIGIENIZAÇÃO, LIMPEZA E EXTRAÇÃO DE DADOS (ETL)
# ==============================================================================

def limpar_documento(doc):
    """Remove caracteres especiais, espaços e zeros à esquerda."""
    if pd.isna(doc):
        return ""
    doc_str = str(doc).strip()
    doc_str = re.sub(r'\.0$', '', doc_str)
    doc_str = re.sub(r'[^a-zA-Z0-9]', '', doc_str)
    return doc_str.lstrip('0')


def e_documento_invalido(doc_limpo):
    """Identifica documentos que contêm apenas datas, textos genéricos ou valores vazios."""
    if not doc_limpo or len(doc_limpo) < 3:
        return True

    if re.match(r'^\d{8}$', doc_limpo):
        dia_ou_ano = int(doc_limpo[:2])
        if 1 <= dia_ou_ano <= 31 or 1900 <= dia_ou_ano <= 2100:
            return True

    padroes_genericos = [r'PIX', r'RECEBIDO', r'TRANSFERENCIA', r'TED', r'DOC', r'PAGTO', r'DEGRANEL', r'DEBITO',
                         r'CREDITO', r'N/A']
    for padrao in padroes_genericos:
        if re.search(padrao, doc_limpo, re.IGNORECASE):
            return True

    return False


def limpar_valor_monetario(val):
    """Converte 'R$ 17.101,65' ou '-1.472,00' em float numérico positivo."""
    if pd.isna(val):
        return 0.0
    val_str = str(val).replace('R$', '').replace(' ', '').strip()
    if ',' in val_str and '.' in val_str:
        val_str = val_str.replace('.', '').replace(',', '.')
    elif ',' in val_str:
        val_str = val_str.replace(',', '.')
    try:
        return abs(float(val_str))
    except ValueError:
        return 0.0


def parse_data_portugues(val):
    """Converte datas em formatos diversos, incluindo abreviações como '3-ago' ou '04-ago-2026'."""
    if pd.isna(val) or not val:
        return pd.NaT

    val_str = str(val).strip().lower()

    meses_pt = {
        'jan': '01', 'fev': '02', 'mar': '03', 'abr': '04',
        'mai': '05', 'jun': '06', 'jul': '07', 'ago': '08',
        'set': '09', 'out': '10', 'nov': '11', 'dez': '12'
    }

    for mes_nome, mes_num in meses_pt.items():
        if mes_nome in val_str:
            val_str = re.sub(rf'\b{mes_nome}\b', mes_num, val_str)
            if re.match(r'^\d{1,2}-\d{2}$', val_str) or re.match(r'^\d{1,2}/\d{2}$', val_str):
                val_str += "-2026"
            break

    try:
        return pd.to_datetime(val_str, dayfirst=True, errors='coerce')
    except Exception:
        return pd.NaT


def carregar_csv_autodetect(arquivo):
    """Lê CSV identificando automaticamente delimitador e codificação."""
    arquivo.seek(0)
    try:
        df = pd.read_csv(arquivo, sep=None, engine='python', encoding='utf-8')
    except Exception:
        arquivo.seek(0)
        df = pd.read_csv(arquivo, sep=None, engine='python', encoding='latin1')

    df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]
    return df


def extrair_texto_e_tabelas_pdf(arquivo_pdf):
    """Extrai transações de PDFs com suporte a tabelas e texto estruturado."""
    linhas_extraida = []
    with pdfplumber.open(arquivo_pdf) as pdf:
        for pagina in pdf.pages:
            tabelas = pagina.extract_tables()
            for tabela in tabelas:
                for linha in tabela:
                    if linha and any(linha):
                        linhas_extraida.append([str(c).strip() if c else '' for c in linha])

            if not tabelas:
                texto = pagina.extract_text()
                if texto:
                    for l in texto.split('\n'):
                        match_valor = re.search(r'(\d{1,3}(?:\.\d{3})*,\d{2})', l)
                        match_data = re.search(r'(\d{1,2}[/-](?:\d{1,2}|[a-zA-Z]{3})[/-]?\d{0,4})', l)
                        if match_valor:
                            dt = match_data.group(1) if match_data else ''
                            val = match_valor.group(1)
                            doc = re.sub(r'[^0-9]', '', l.replace(dt, '').replace(val, ''))
                            linhas_extraida.append([dt, doc, val, l])

    return pd.DataFrame(linhas_extraida)


def encontrar_coluna(df, palavras_chave):
    """Localiza a melhor coluna no DataFrame com base em múltiplos nomes/sinônimos."""
    for col in df.columns:
        col_clean = str(col).upper()
        if any(pc.upper() in col_clean for pc in palavras_chave):
            return col
    return None


# ==============================================================================
# MÓDULO A.1: PRÉ-PROCESSAMENTO E CRUZAMENTO TRIPLO (FINANCEIRO x EXTRATO)
# ==============================================================================

def fundir_financeiro_com_extrato(df_fin, df_ext, tolerancia=0.05):
    """
    Realiza o batimento entre a Planilha do Financeiro e o Extrato Bancário.
    Usa Data e Valor como chaves para localizar o Número do Documento no Extrato.
    Evita duplicação de pagamentos.
    """
    if df_fin is None or df_fin.empty:
        return df_ext

    if df_ext is None or df_ext.empty:
        return df_fin

    pagamentos_unificados = []

    for _, row_fin in df_fin.iterrows():
        dt_fin = row_fin.get('Data_Parsed', pd.NaT)
        val_fin = float(row_fin.get('Valor_Pago', 0.0))
        doc_fin = str(row_fin.get('Doc_Limpo', ''))
        obs_fin = str(row_fin.get('Observacao_Origem', ''))

        # Tenta localizar o registro correspondente no extrato bancário (PROCV por Data e Valor)
        match_ext = df_ext[
            (df_ext['Data_Parsed'] == dt_fin) &
            (np.isclose(df_ext['Valor_Pago'], val_fin, atol=tolerancia))
            ]

        doc_enriquecido = doc_fin
        origem = obs_fin

        if not match_ext.empty:
            match_row = match_ext.iloc[0]
            doc_ext = str(match_row.get('Doc_Limpo', ''))
            if doc_ext and not e_documento_invalido(doc_ext):
                doc_enriquecido = doc_ext
            origem += f" | Reconciliado c/ Extrato ({match_row.get('Doc_Original', '')})"

        pagamentos_unificados.append({
            'Data_Parsed': dt_fin,
            'Doc_Original': row_fin.get('Doc_Original', ''),
            'Doc_Limpo': doc_enriquecido,
            'Valor_Pago': val_fin,
            'Tipo_Operacao': row_fin.get('Tipo_Operacao', 'PIX'),
            'Observacao_Origem': origem
        })

    # Adiciona registros do Extrato que porventura não constavam na planilha do financeiro
    for _, row_ext in df_ext.iterrows():
        dt_ext = row_ext.get('Data_Parsed', pd.NaT)
        val_ext = float(row_ext.get('Valor_Pago', 0.0))

        match_fin = df_fin[
            (df_fin['Data_Parsed'] == dt_ext) &
            (np.isclose(df_fin['Valor_Pago'], val_ext, atol=tolerancia))
            ]
        if match_fin.empty:
            pagamentos_unificados.append({
                'Data_Parsed': dt_ext,
                'Doc_Original': row_ext.get('Doc_Original', ''),
                'Doc_Limpo': row_ext.get('Doc_Limpo', ''),
                'Valor_Pago': val_ext,
                'Tipo_Operacao': row_ext.get('Tipo_Operacao', 'PIX'),
                'Observacao_Origem': "Exclusivo Extrato Bancário"
            })

    return pd.DataFrame(pagamentos_unificados)


# ==============================================================================
# MÓDULO B: MOTOR DE CONCILIAÇÃO INTELIGENTE (WATERFALL MATCHING)
# ==============================================================================

def executar_waterfall_matching(df_pagamentos, df_faturamento, janela_dias=10, tolerancia=0.05):
    resultados = []
    df_fat = df_faturamento.copy()

    col_os = encontrar_coluna(df_fat, ['OS', 'ORDEM', 'SERVICO']) or df_fat.columns[0]
    col_nf = encontrar_coluna(df_fat, ['NF-E', 'NFS-E', 'NUMERO DA NF', 'NF', 'NOTA', 'DOCUMENTO']) or df_fat.columns[1]
    col_titulo = encontrar_coluna(df_fat, ['TITULO', 'TÍTULO', 'PARCELA', 'ID_TITULO']) or col_nf
    col_val_tit = encontrar_coluna(df_fat,
                                   ['VALOR_TITULO', 'VALOR_PARCELA', 'VALOR TITULO', 'VALOR DO TITULO', 'VALOR']) or \
                  df_fat.columns[-1]
    col_val_tot_nf = encontrar_coluna(df_fat, ['VALOR_TOTAL_NF', 'TOTAL_NF', 'TOTAL DA NOTA', 'VALOR TOTAL'])
    col_emissao = encontrar_coluna(df_fat, ['EMISSAO', 'EMISSÃO', 'DATA_EMISSAO', 'DATA DE EMISSAO', 'DATA'])
    col_status = encontrar_coluna(df_fat, ['STATUS', 'SITUAÇÃO', 'SITUACAO', 'STATUS_TITULO', 'STATUS TITULO'])

    if col_val_tit in df_fat.columns:
        df_fat[col_val_tit] = df_fat[col_val_tit].apply(limpar_valor_monetario)
    if col_val_tot_nf and col_val_tot_nf in df_fat.columns:
        df_fat[col_val_tot_nf] = df_fat[col_val_tot_nf].apply(limpar_valor_monetario)

    nf_group = df_fat.groupby(col_nf).agg({
        col_val_tit: 'sum',
        col_os: 'first',
        col_emissao: 'min' if col_emissao else 'first',
        col_titulo: list
    }).reset_index()

    os_group = df_fat.groupby(col_os).agg({
        col_val_tit: 'sum',
        col_nf: lambda x: list(set(x)),
        col_emissao: 'min' if col_emissao else 'first'
    }).reset_index()

    for _, pag in df_pagamentos.iterrows():
        doc_ext = str(pag.get('Doc_Limpo', ''))
        val_ext = float(pag.get('Valor_Pago', 0.0))
        dt_ext = pag.get('Data_Parsed', pd.NaT)
        tp_op = str(pag.get('Tipo_Operacao', 'PIX')).upper()
        obs_origem = str(pag.get('Observacao_Origem', ''))

        status_match = "NAO_LOCALIZADO"
        regra_match = "NENHUMA"
        detalhes_match = {}

        is_doc_invalido = e_documento_invalido(doc_ext)

        # NÍVEL 1: Chave Exata (Número da NF/Título + Valor Exato)
        if not is_doc_invalido and val_ext > 0:
            match_l1 = df_fat[
                ((df_fat[col_nf].astype(str).apply(limpar_documento) == doc_ext) |
                 (df_fat[col_titulo].astype(str).apply(limpar_documento) == doc_ext)) &
                (np.isclose(df_fat[col_val_tit], val_ext, atol=tolerancia))
                ]
            if not match_l1.empty:
                row = match_l1.iloc[0]
                status_match = "CONCILIADO_EXATO"
                regra_match = "NIVEL_1_CHAVE_EXATA"
                detalhes_match = {
                    'ID_OS': row[col_os],
                    'Numero_NF': row[col_nf],
                    'ID_Titulo': row[col_titulo],
                    'Data_Emissao_NF': row[col_emissao] if col_emissao else pd.NaT,
                    'Status_Titulo': row[col_status] if col_status and col_status in row else 'N/A'
                }

        # NÍVEL 2: Antecipação Total de Nota Parcelada
        if status_match == "NAO_LOCALIZADO" and not is_doc_invalido and val_ext > 0:
            match_l2 = nf_group[
                (nf_group[col_nf].astype(str).apply(limpar_documento) == doc_ext) &
                (np.isclose(nf_group[col_val_tit], val_ext, atol=tolerancia))
                ]
            if not match_l2.empty:
                row = match_l2.iloc[0]
                status_match = "ANTECIPACAO_TOTAL_PIX"
                regra_match = "NIVEL_2_ANTECIPACAO_TOTAL"
                detalhes_match = {
                    'ID_OS': row[col_os],
                    'Numero_NF': row[col_nf],
                    'ID_Titulo': f"TODOS_TITULOS ({len(row[col_titulo])} parcelas)",
                    'Data_Emissao_NF': row[col_emissao] if col_emissao else pd.NaT,
                    'Status_Titulo': 'PARCELAS_QUITADAS'
                }

        # NÍVEL 3: Busca Alternativa por Valor Exato + Janela Temporal
        if status_match == "NAO_LOCALIZADO" and val_ext > 0:
            df_fat_temp = df_fat.copy()
            if pd.notna(dt_ext) and col_emissao:
                df_fat_temp['Diff_Dias'] = (df_fat_temp[col_emissao] - dt_ext).abs().dt.days
                df_fat_temp = df_fat_temp[df_fat_temp['Diff_Dias'] <= janela_dias]

            match_l3 = df_fat_temp[np.isclose(df_fat_temp[col_val_tit], val_ext, atol=tolerancia)]
            if not match_l3.empty:
                row = match_l3.iloc[0]
                status_match = "CONCILIADO_POR_VALOR"
                regra_match = "NIVEL_3_BUSCA_VALOR_TEMPORAL"
                detalhes_match = {
                    'ID_OS': row[col_os],
                    'Numero_NF': row[col_nf],
                    'ID_Titulo': row[col_titulo],
                    'Data_Emissao_NF': row[col_emissao] if col_emissao else pd.NaT,
                    'Status_Titulo': row[col_status] if col_status and col_status in row else 'N/A'
                }
            else:
                match_l3_nf = nf_group[np.isclose(nf_group[col_val_tit], val_ext, atol=tolerancia)]
                if pd.notna(dt_ext) and col_emissao:
                    match_l3_nf = match_l3_nf[(match_l3_nf[col_emissao] - dt_ext).abs().dt.days <= janela_dias]
                if not match_l3_nf.empty:
                    row = match_l3_nf.iloc[0]
                    status_match = "ANTECIPACAO_TOTAL_PIX"
                    regra_match = "NIVEL_3_ANTECIPACAO_POR_VALOR"
                    detalhes_match = {
                        'ID_OS': row[col_os],
                        'Numero_NF': row[col_nf],
                        'ID_Titulo': "TODOS_TITULOS_NF",
                        'Data_Emissao_NF': row[col_emissao] if col_emissao else pd.NaT,
                        'Status_Titulo': 'PARCELAS_QUITADAS'
                    }

        # NÍVEL 4: Conciliação OS Multissegmentada
        if status_match == "NAO_LOCALIZADO" and val_ext > 0:
            match_l4 = os_group[np.isclose(os_group[col_val_tit], val_ext, atol=tolerancia)]
            if pd.notna(dt_ext) and col_emissao:
                match_l4 = match_l4[(match_l4[col_emissao] - dt_ext).abs().dt.days <= janela_dias * 2]
            if not match_l4.empty:
                row = match_l4.iloc[0]
                status_match = "CONCILIADO_OS_MULTISSEGMENTADA"
                regra_match = "NIVEL_4_SOMA_NFS_OS"
                detalhes_match = {
                    'ID_OS': row[col_os],
                    'Numero_NF': f"NFs: {row[col_nf]}",
                    'ID_Titulo': "MULTIPLOS_TITULOS_OS",
                    'Data_Emissao_NF': row[col_emissao] if col_emissao else pd.NaT,
                    'Status_Titulo': 'MULTIPLIS_NFS'
                }

        resultados.append({
            'Data_Pagamento': dt_ext,
            'Doc_Extrato_Original': pag.get('Doc_Original', ''),
            'Doc_Extrato_Limpo': doc_ext,
            'Valor_Pago': val_ext,
            'Tipo_Operacao': tp_op,
            'Observacao_Origem': obs_origem,
            'Status_Conciliacao': status_match,
            'Regra_Aplicada': regra_match,
            'ID_OS': detalhes_match.get('ID_OS', 'N/A'),
            'Numero_NF': detalhes_match.get('Numero_NF', 'N/A'),
            'ID_Titulo': detalhes_match.get('ID_Titulo', 'N/A'),
            'Data_Emissao_NF': detalhes_match.get('Data_Emissao_NF', pd.NaT),
            'Status_Titulo_Truckpag': detalhes_match.get('Status_Titulo', 'N/A')
        })

    return pd.DataFrame(resultados)


# ==============================================================================
# MÓDULO C: AUDITORIA E TRAVAS DE SEGURANÇA
# ==============================================================================

def aplicar_auditoria_e_travas(df_conciliado, data_corte_param, col_status=None):
    df_audit = df_conciliado.copy()
    dt_corte = pd.to_datetime(data_corte_param)

    # 1. Infração: Emitido a partir de 17/06 e pago via PIX
    cond_infracao_corte = (
            (df_audit['Data_Emissao_NF'] >= dt_corte) &
            (df_audit['Tipo_Operacao'].astype(str).str.upper().str.contains('PIX', na=False)) &
            (df_audit['Status_Conciliacao'] != 'NAO_LOCALIZADO')
    )

    df_audit['Alerta_Infracao_Corte_1706'] = cond_infracao_corte
    df_audit.loc[cond_infracao_corte, 'Status_Conciliacao'] = 'DIVERGENCIA_CORTE_1706'

    # 2. Regra FIFO Refinada: Pagamento efetuado antes da emissão da NF E (se existir) Status Baixado
    cond_erro_fifo = (
            (df_audit['Data_Pagamento'] < df_audit['Data_Emissao_NF']) &
            (df_audit['Status_Conciliacao'] != 'NAO_LOCALIZADO')
    )

    if col_status and col_status in df_audit.columns:
        status_validos = ['BAIXADO', 'PAGO', 'QUITADO', 'BAIXADO_AUTOMATICO', 'PARCELAS_QUITADAS']
        cond_erro_fifo = cond_erro_fifo & (
            df_audit[col_status].astype(str).str.upper().isin(status_validos)
        )

    df_audit['Alerta_Erro_FIFO_Lancamento_Tardio'] = cond_erro_fifo
    df_audit.loc[cond_erro_fifo & (
        ~df_audit.get('Alerta_Infracao_Corte_1706', False)), 'Status_Conciliacao'] = 'POSSIVEL_BAIXA_INDEVIDA_FIFO'

    return df_audit


# ==============================================================================
# MÓDULO D: RELATÓRIOS E WHATSAPP PAYLOAD
# ==============================================================================

def gerar_payload_whatsapp(df_audit):
    df_pix = df_audit[
        (df_audit['Status_Conciliacao'].isin([
            'CONCILIADO_EXATO', 'ANTECIPACAO_TOTAL_PIX', 'CONCILIADO_POR_VALOR', 'CONCILIADO_OS_MULTISSEGMENTADA'
        ]))
    ]

    if df_pix.empty:
        return "Nenhuma baixa via PIX pendente identificada para envio manual."

    linhas_msg = ["*SOLICITAÇÃO DE BAIXA BANCÁRIA - DEGRANEL (PIX)*\n"]
    for idx, row in df_pix.reset_index(drop=True).iterrows():
        dt_str = row['Data_Pagamento'].strftime('%d/%m/%Y') if pd.notna(row['Data_Pagamento']) else 'N/A'
        msg = (
            f"🔹 *Item {idx + 1}*\n"
            f"• *Data Pagto:* {dt_str}\n"
            f"• *Valor:* R$ {row['Valor_Pago']:,.2f}\n"
            f"• *NF / Doc:* {row['Numero_NF']}\n"
            f"• *Título/Parcela:* {row['ID_Titulo']}\n"
            f"• *OS:* {row['ID_OS']}\n"
            f"• *Regra Match:* {row['Regra_Aplicada']}\n"
        )
        linhas_msg.append(msg)

    return "\n".join(linhas_msg)


def gerar_excel_download(df_audit):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_audit.to_excel(writer, sheet_name='Geral_Conciliacao', index=False)
        for status in df_audit['Status_Conciliacao'].unique():
            df_sub = df_audit[df_audit['Status_Conciliacao'] == status]
            sheet_title = str(status)[:30]
            df_sub.to_excel(writer, sheet_name=sheet_title, index=False)
    output.seek(0)
    return output


# ==============================================================================
# INTERFACE WEB STREAMLIT
# ==============================================================================

st.title("🚛 Sistema de Conciliação Inteligente - Truckpag & Degranel")
st.markdown("Cruzamento em cascata (Extratos + Planilha Financeira vs. Faturamento Truckpag).")

# 1. PARÂMETROS ADICIONADOS NA BARRA LATERAL
st.sidebar.header("⚙️ Parâmetros do Sistema")
param_data_inicio = st.sidebar.date_input("Analisar a partir de:", datetime(2026, 8, 3))
param_data_corte = st.sidebar.date_input("Data de Corte (Boleto Obrigatório)", datetime(2026, 6, 17))
param_janela_dias = st.sidebar.slider("Janela Temporal de Busca (Dias)", min_value=1, max_value=30, value=10)
param_tolerancia = st.sidebar.number_input("Tolerância de Valor (R$)", min_value=0.0, max_value=10.0, value=0.05,
                                           step=0.05)

st.sidebar.divider()
st.sidebar.info("Ajuste a tolerância em R$ caso o cliente tenha efetuado pagamentos com variações de centavos.")

st.subheader("📁 Carga dos Arquivos para Conciliação")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**1. Faturamento Truckpag (CSV)**")
    file_fat = st.file_uploader("Upload Consulta de Faturamento", type=["csv"])

with col2:
    st.markdown("**2. Extratos Bancários Degranel (PDF/CSV)**")
    files_ext = st.file_uploader("Upload Extrato(s) Bancário(s)", type=["pdf", "csv"], accept_multiple_files=True)

with col3:
    st.markdown("**3. Planilha do Financeiro (CSV/PDF)**")
    file_fin = st.file_uploader("Upload Planilha Financeira", type=["csv", "pdf"])

st.divider()

if st.button("⚡ Processar Conciliação e Auditoria", type="primary"):
    if not file_fat or (not files_ext and not file_fin):
        st.error(
            "⚠️ Envie o arquivo de Faturamento e ao menos uma fonte de pagamento (Extrato ou Planilha Financeira).")
    else:
        with st.spinner("Processando ETL, cruzamento triplo e motor em cascata..."):

            # 1. CARGA FATURAMENTO
            df_fat = carregar_csv_autodetect(file_fat)
            col_status_fat = encontrar_coluna(df_fat, ['STATUS', 'SITUAÇÃO', 'SITUACAO', 'STATUS_TITULO'])

            # 2. CARGA EXTRATOS BANCÁRIOS
            df_ext_tot = None
            if files_ext:
                lista_ext = []
                for f_ext in files_ext:
                    if f_ext.name.lower().endswith('.pdf'):
                        df_pdf = extrair_texto_e_tabelas_pdf(f_ext)
                        if not df_pdf.empty:
                            col_d = 0 if 0 in df_pdf.columns else df_pdf.columns[0]
                            col_doc = 1 if 1 in df_pdf.columns else df_pdf.columns[0]
                            col_v = 2 if 2 in df_pdf.columns else df_pdf.columns[-1]
                            for _, r in df_pdf.iterrows():
                                lista_ext.append({
                                    'Data_Parsed': parse_data_portugues(r[col_d]),
                                    'Doc_Original': r[col_doc],
                                    'Doc_Limpo': limpar_documento(r[col_doc]),
                                    'Valor_Pago': limpar_valor_monetario(r[col_v]),
                                    'Tipo_Operacao': 'PIX',
                                    'Observacao_Origem': f"Extrato PDF ({f_ext.name})"
                                })
                    else:
                        df_csv_ext = carregar_csv_autodetect(f_ext)
                        c_dt = encontrar_coluna(df_csv_ext, ['DATA', 'LANÇAMENTO', 'TRANSACAO']) or df_csv_ext.columns[
                            0]
                        c_doc = encontrar_coluna(df_csv_ext, ['DOC', 'NUMERO', 'HISTORICO']) or df_csv_ext.columns[0]
                        c_val = encontrar_coluna(df_csv_ext, ['VALOR', 'CREDITO', 'ENTRADA']) or df_csv_ext.columns[-1]
                        for _, r in df_csv_ext.iterrows():
                            lista_ext.append({
                                'Data_Parsed': parse_data_portugues(r[c_dt]),
                                'Doc_Original': r[c_doc],
                                'Doc_Limpo': limpar_documento(r[c_doc]),
                                'Valor_Pago': limpar_valor_monetario(r[c_val]),
                                'Tipo_Operacao': 'PIX',
                                'Observacao_Origem': f"Extrato CSV ({f_ext.name})"
                            })
                df_ext_tot = pd.DataFrame(lista_ext)

            # 3. CARGA PLANILHA FINANCEIRO
            df_fin_tot = None
            if file_fin:
                if file_fin.name.lower().endswith('.csv'):
                    df_fin_raw = carregar_csv_autodetect(file_fin)
                else:
                    df_fin_raw = extrair_texto_e_tabelas_pdf(file_fin)

                if not df_fin_raw.empty:
                    c_dt_fin = encontrar_coluna(df_fin_raw, ['DATA PAG', 'DATA', 'LANÇAMENTO']) or df_fin_raw.columns[0]
                    c_doc_fin = encontrar_coluna(df_fin_raw, ['TITULO', 'TÍTULO', 'NF', 'DOCUMENTO']) or \
                                df_fin_raw.columns[0]
                    c_val_fin = encontrar_coluna(df_fin_raw, ['VALOR', 'VALOR_PAGO']) or df_fin_raw.columns[1]
                    c_obs_fin = encontrar_coluna(df_fin_raw, ['COMO DEVERIA', 'PAGO VIA', 'STATUS'])

                    lista_fin = []
                    for _, r in df_fin_raw.iterrows():
                        obs = str(r[c_obs_fin]) if c_obs_fin and c_obs_fin in r else "Planilha Financeiro"
                        lista_fin.append({
                            'Data_Parsed': parse_data_portugues(r[c_dt_fin]),
                            'Doc_Original': r[c_doc_fin],
                            'Doc_Limpo': limpar_documento(r[c_doc_fin]),
                            'Valor_Pago': limpar_valor_monetario(r[c_val_fin]),
                            'Tipo_Operacao': 'PIX' if 'PIX' in obs.upper() else 'BOLETO/OUTRO',
                            'Observacao_Origem': f"Planilha Fin: {obs}"
                        })
                    df_fin_tot = pd.DataFrame(lista_fin)

            # --- APLICANDO O CRUZAMENTO TRIPLO (FINANCEIRO x EXTRATO) ---
            df_pagamentos_unificados = fundir_financeiro_com_extrato(
                df_fin=df_fin_tot,
                df_ext=df_ext_tot,
                tolerancia=param_tolerancia
            )

            # --- APLICANDO O FILTRO OBRIGATÓRIO DE DATA INICIAL (03/08) ---
            if not df_pagamentos_unificados.empty:
                dt_inicio_dt = pd.to_datetime(param_data_inicio)
                df_pagamentos_unificados = df_pagamentos_unificados[
                    df_pagamentos_unificados['Data_Parsed'] >= dt_inicio_dt
                    ].copy()

            # Normalização das datas de emissão do faturamento Truckpag
            col_emissao_fat = encontrar_coluna(df_fat, ['EMISSAO', 'EMISSÃO', 'DATA_EMISSAO', 'DATA DE EMISSAO'])
            if col_emissao_fat and col_emissao_fat in df_fat.columns:
                df_fat[col_emissao_fat] = df_fat[col_emissao_fat].apply(parse_data_portugues)

            # --- PAINEL DIAGNÓSTICO ---
            with st.expander("🔍 **Inspecionar Leitura e Cruzamento Triplo (Debug)**", expanded=False):
                d1, d2 = st.columns(2)
                with d1:
                    st.write(f"**Faturamento Truckpag:** {len(df_fat)} linhas")
                    st.dataframe(df_fat.head(3), use_container_width=True)
                with d2:
                    st.write(
                        f"**Pagamentos Reconciliados (A partir de {param_data_inicio.strftime('%d/%m/%Y')}):** {len(df_pagamentos_unificados)} linhas")
                    st.dataframe(df_pagamentos_unificados.head(5), use_container_width=True)

            # EXECUÇÃO DO WATERFALL MATCHING
            df_conciliado = executar_waterfall_matching(
                df_pagamentos=df_pagamentos_unificados,
                df_faturamento=df_fat,
                janela_dias=param_janela_dias,
                tolerancia=param_tolerancia
            )

            # APLICAÇÃO DE AUDITORIA E TRAVAS REFINADAS
            df_resultado_final = aplicar_auditoria_e_travas(
                df_conciliado=df_conciliado,
                data_corte_param=param_data_corte,
                col_status=col_status_fat
            )

            st.success(
                f"✅ Processamento e auditoria concluídos com sucesso! {len(df_resultado_final)} transações analisadas.")

            # EXIBIÇÃO DOS RESULTADOS POR CATEGORIA
            st.subheader("📊 Painel de Resultados por Categoria")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Conciliado Exato",
                      len(df_resultado_final[df_resultado_final['Status_Conciliacao'] == 'CONCILIADO_EXATO']))
            m2.metric("Antecipação PIX",
                      len(df_resultado_final[df_resultado_final['Status_Conciliacao'] == 'ANTECIPACAO_TOTAL_PIX']))
            m3.metric("Infração Corte 17/06",
                      len(df_resultado_final[df_resultado_final['Status_Conciliacao'] == 'DIVERGENCIA_CORTE_1706']))
            m4.metric("Erro FIFO (Lanç. Tardio)", len(
                df_resultado_final[df_resultado_final['Status_Conciliacao'] == 'POSSIVEL_BAIXA_INDEVIDA_FIFO']))
            m5.metric("Não Localizado",
                      len(df_resultado_final[df_resultado_final['Status_Conciliacao'] == 'NAO_LOCALIZADO']))

            st.divider()

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📋 Resumo Geral",
                "🚨 Infrações Corte 17/06",
                "⚠️ Suspeitas Erro FIFO",
                "⚡ Antecipações PIX",
                "❌ Não Localizados",
                "💬 Payload WhatsApp"
            ])

            with tab1:
                st.dataframe(df_resultado_final, use_container_width=True)

            with tab2:
                st.write("### Pagamentos PIX em notas emitidas a partir de 17/06 (Obrigatório Boleto)")
                st.dataframe(df_resultado_final[df_resultado_final['Status_Conciliacao'] == 'DIVERGENCIA_CORTE_1706'],
                             use_container_width=True)

            with tab3:
                st.write(
                    "### Pagamentos efetuados antes da emissão da NF (Erro FIFO / Baixa indevida em título antigo)")
                st.dataframe(
                    df_resultado_final[df_resultado_final['Status_Conciliacao'] == 'POSSIVEL_BAIXA_INDEVIDA_FIFO'],
                    use_container_width=True)

            with tab4:
                st.write("### Notas parceladas quitadas integralmente via PIX de uma só vez")
                st.dataframe(df_resultado_final[df_resultado_final['Status_Conciliacao'] == 'ANTECIPACAO_TOTAL_PIX'],
                             use_container_width=True)

            with tab5:
                st.write("### Pagamentos sem correspondência encontrada no Faturamento Truckpag")
                st.dataframe(df_resultado_final[df_resultado_final['Status_Conciliacao'] == 'NAO_LOCALIZADO'],
                             use_container_width=True)

            with tab6:
                st.write("### Payload Formatado para Envio ao Grupo do WhatsApp")
                st.code(gerar_payload_whatsapp(df_resultado_final), language="text")

            excel_bytes = gerar_excel_download(df_resultado_final)
            st.download_button(
                label="📥 Baixar Relatório Completo em Excel (Multi-Abas)",
                data=excel_bytes,
                file_name=f"Conciliacao_Truckpag_Degranel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )