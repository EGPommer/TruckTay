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
    page_title="Sistema de Conciliação - DGranel / TruckPag",
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

    padroes_genericos = [r'PIX', r'RECEBIDO', r'TRANSFERENCIA', r'TED', r'DOC', r'PAGTO', r'DGRANEL', r'DEBITO',
                         r'CREDITO', r'N/A', r'-']
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
    if pd.isna(val) or not val or str(val).strip() in ['-', '--', 'N/A', 'nan', 'None']:
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
    """Localiza a melhor coluna no DataFrame com base em múltiplos nomes/sinónimos."""
    for col in df.columns:
        col_clean = str(col).upper()
        if any(pc.upper() in col_clean for pc in palavras_chave):
            return col
    return None


# ==============================================================================
# MÓDULO A.1: RECONCILIAÇÃO PAGAMENTOS PIX x EXTRATO BANCÁRIO
# ==============================================================================

def fundir_pix_com_extrato(df_pix, df_ext, tolerancia=0.05):
    """Cruza a listagem de Pagamentos PIX com os Extratos Bancários para obter detalhes adicionais."""
    if df_pix is None or df_pix.empty:
        return df_ext if df_ext is not None else pd.DataFrame()

    if df_ext is None or df_ext.empty:
        return df_pix

    pagamentos_unificados = []

    for _, row_pix in df_pix.iterrows():
        dt_pix = row_pix.get('Data_Parsed', pd.NaT)
        val_pix = float(row_pix.get('Valor_Pago', 0.0))
        doc_pix = str(row_pix.get('Doc_Limpo', ''))
        obs_pix = str(row_pix.get('Observacao_Origem', ''))

        match_ext = df_ext[
            (df_ext['Data_Parsed'] == dt_pix) &
            (np.isclose(df_ext['Valor_Pago'], val_pix, atol=tolerancia))
            ]

        doc_enriquecido = doc_pix
        origem = obs_pix

        if not match_ext.empty:
            match_row = match_ext.iloc[0]
            doc_ext = str(match_row.get('Doc_Limpo', ''))
            if doc_ext and not e_documento_invalido(doc_ext):
                doc_enriquecido = doc_ext
            origem += f" | Confirmado no Extrato ({match_row.get('Doc_Original', '')})"

        pagamentos_unificados.append({
            'Data_Parsed': dt_pix,
            'Doc_Original': row_pix.get('Doc_Original', ''),
            'Doc_Limpo': doc_enriquecido,
            'Valor_Pago': val_pix,
            'Tipo_Operacao': row_pix.get('Tipo_Operacao', 'PIX'),
            'Observacao_Origem': origem
        })

    for _, row_ext in df_ext.iterrows():
        dt_ext = row_ext.get('Data_Parsed', pd.NaT)
        val_ext = float(row_ext.get('Valor_Pago', 0.0))

        match_pix = df_pix[
            (df_pix['Data_Parsed'] == dt_ext) &
            (np.isclose(df_pix['Valor_Pago'], val_ext, atol=tolerancia))
            ]
        if match_pix.empty:
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

    # Mapeamento estendido de colunas da Gestão TruckPag
    col_os = encontrar_coluna(df_fat, ['OS', 'ORDEM', 'SERVICO']) or df_fat.columns[0]
    col_nf = encontrar_coluna(df_fat, ['NF-E', 'NFS-E', 'NUMERO DA NF', 'NF', 'NOTA', 'DOCUMENTO']) or df_fat.columns[1]
    col_titulo = encontrar_coluna(df_fat, ['TITULO', 'TÍTULO', 'PARCELA', 'ID_TITULO']) or col_nf
    col_val_tit = encontrar_coluna(df_fat,
                                   ['VALOR_TITULO', 'VALOR_PARCELA', 'VALOR TITULO', 'VALOR DO TITULO', 'VALOR']) or \
                  df_fat.columns[-1]
    col_val_tot_nf = encontrar_coluna(df_fat, ['VALOR_TOTAL_NF', 'TOTAL_NF', 'TOTAL DA NOTA', 'VALOR TOTAL'])
    col_emissao = encontrar_coluna(df_fat, ['EMISSAO', 'EMISSÃO', 'DATA_EMISSAO', 'DATA DE EMISSAO'])
    col_vencimento = encontrar_coluna(df_fat, ['VENCIMENTO', 'DATA VENCIMENTO', 'DT_VENC', 'DATA DE VENCIMENTO'])
    col_pagto_gestao = encontrar_coluna(df_fat, ['DATA PAGAMENTO', 'DATA_PAGAMENTO', 'DATA PAGTO', 'PAGAMENTO'])
    col_status = encontrar_coluna(df_fat, ['STATUS', 'SITUAÇÃO', 'SITUACAO', 'STATUS_TITULO', 'STATUS TITULO'])

    # Limpeza numérica
    if col_val_tit in df_fat.columns:
        df_fat[col_val_tit] = df_fat[col_val_tit].apply(limpar_valor_monetario)
    if col_val_tot_nf and col_val_tot_nf in df_fat.columns:
        df_fat[col_val_tot_nf] = df_fat[col_val_tot_nf].apply(limpar_valor_monetario)

    # Grupos de agregação
    nf_group = df_fat.groupby(col_nf).agg({
        col_val_tit: 'sum',
        col_os: 'first',
        col_emissao: 'min' if col_emissao else 'first',
        col_vencimento: 'min' if col_vencimento else 'first',
        col_titulo: list
    }).reset_index()

    os_group = df_fat.groupby(col_os).agg({
        col_val_tit: 'sum',
        col_nf: lambda x: list(set(x)),
        col_emissao: 'min' if col_emissao else 'first',
        col_vencimento: 'min' if col_vencimento else 'first'
    }).reset_index()

    for _, pag in df_pagamentos.iterrows():
        doc_ext = str(pag.get('Doc_Limpo', ''))
        val_ext = float(pag.get('Valor_Pago', 0.0))
        dt_ext = pag.get('Data_Parsed', pd.NaT)
        tp_op = str(pag.get('Tipo_Operacao', 'PIX')).upper()
        obs_origem = str(pag.get('Observacao_Origem', ''))

        status_match = "NAO_LOCALIZADO"
        regra_match = "NENHUMA"
        detalhes = {}

        is_doc_invalido = e_documento_invalido(doc_ext)

        # NÍVEL 1: Chave Exata (Doc/NF/Título + Valor Exato)
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
                detalhes = {
                    'ID_OS': row[col_os],
                    'Numero_NF': row[col_nf],
                    'ID_Titulo': row[col_titulo],
                    'Data_Emissao_NF': row[col_emissao] if col_emissao else pd.NaT,
                    'Data_Vencimento': row[col_vencimento] if col_vencimento else pd.NaT,
                    'Data_Pagamento_Gestao': row[col_pagto_gestao] if col_pagto_gestao else 'N/A',
                    'Valor_Titulo': row[col_val_tit],
                    'Status_Titulo_Truckpag': row[col_status] if col_status and col_status in row else 'N/A'
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
                detalhes = {
                    'ID_OS': row[col_os],
                    'Numero_NF': row[col_nf],
                    'ID_Titulo': f"TODOS_TITULOS ({len(row[col_titulo])} parcelas)",
                    'Data_Emissao_NF': row[col_emissao] if col_emissao else pd.NaT,
                    'Data_Vencimento': row[col_vencimento] if col_vencimento else pd.NaT,
                    'Data_Pagamento_Gestao': 'BAIXA_ANTECIPADA',
                    'Valor_Titulo': row[col_val_tit],
                    'Status_Titulo_Truckpag': 'PARCELAS_QUITADAS'
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
                detalhes = {
                    'ID_OS': row[col_os],
                    'Numero_NF': row[col_nf],
                    'ID_Titulo': row[col_titulo],
                    'Data_Emissao_NF': row[col_emissao] if col_emissao else pd.NaT,
                    'Data_Vencimento': row[col_vencimento] if col_vencimento else pd.NaT,
                    'Data_Pagamento_Gestao': row[col_pagto_gestao] if col_pagto_gestao else 'N/A',
                    'Valor_Titulo': row[col_val_tit],
                    'Status_Titulo_Truckpag': row[col_status] if col_status and col_status in row else 'N/A'
                }
            else:
                match_l3_nf = nf_group[np.isclose(nf_group[col_val_tit], val_ext, atol=tolerancia)]
                if pd.notna(dt_ext) and col_emissao:
                    match_l3_nf = match_l3_nf[(match_l3_nf[col_emissao] - dt_ext).abs().dt.days <= janela_dias]
                if not match_l3_nf.empty:
                    row = match_l3_nf.iloc[0]
                    status_match = "ANTECIPACAO_TOTAL_PIX"
                    regra_match = "NIVEL_3_ANTECIPACAO_POR_VALOR"
                    detalhes = {
                        'ID_OS': row[col_os],
                        'Numero_NF': row[col_nf],
                        'ID_Titulo': "TODOS_TITULOS_NF",
                        'Data_Emissao_NF': row[col_emissao] if col_emissao else pd.NaT,
                        'Data_Vencimento': row[col_vencimento] if col_vencimento else pd.NaT,
                        'Data_Pagamento_Gestao': 'BAIXA_ANTECIPADA',
                        'Valor_Titulo': row[col_val_tit],
                        'Status_Titulo_Truckpag': 'PARCELAS_QUITADAS'
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
                detalhes = {
                    'ID_OS': row[col_os],
                    'Numero_NF': f"NFs: {row[col_nf]}",
                    'ID_Titulo': "MULTIPLOS_TITULOS_OS",
                    'Data_Emissao_NF': row[col_emissao] if col_emissao else pd.NaT,
                    'Data_Vencimento': row[col_vencimento] if col_vencimento else pd.NaT,
                    'Data_Pagamento_Gestao': 'SOMA_NFS',
                    'Valor_Titulo': row[col_val_tit],
                    'Status_Titulo_Truckpag': 'MULTIPLAS_NFS'
                }

        # Cálculo de valores e estado de pagamento
        val_titulo = float(detalhes.get('Valor_Titulo', 0.0))
        valor_em_aberto = max(0.0, val_titulo - val_ext) if val_titulo > 0 else 0.0

        # Identificação de Pagamento Parcial
        if val_titulo > 0 and 0 < val_ext < (val_titulo - tolerancia):
            tipo_pagamento = "PAGAMENTO_PARCIAL"
        elif val_titulo > 0 and np.isclose(val_ext, val_titulo, atol=tolerancia):
            tipo_pagamento = "PAGAMENTO_INTEGRAL"
        else:
            tipo_pagamento = "INDETERMINADO"

        # Verificação da Data de Pagamento na Gestão (vazia ou '-' = NÃO PAGO)
        dt_pagto_gestao_raw = str(detalhes.get('Data_Pagamento_Gestao', '')).strip()
        if dt_pagto_gestao_raw in ['', '-', '--', 'nan', 'None', 'NaT']:
            estado_liquidacao = "EM ABERTO (NÃO PAGO)"
        else:
            estado_liquidacao = "BAIXADO (PAGO)"

        resultados.append({
            'Data_Pagamento_DGranel': dt_ext,
            'Data_Vencimento': detalhes.get('Data_Vencimento', pd.NaT),
            'Data_Pagamento_Gestao': dt_pagto_gestao_raw,
            'Doc_Extrato_Original': pag.get('Doc_Original', ''),
            'Doc_Extrato_Limpo': doc_ext,
            'Valor_Titulo': val_titulo,
            'Valor_Pago_DGranel': val_ext,
            'Valor_Em_Aberto': valor_em_aberto,
            'Tipo_Pagamento': tipo_pagamento,
            'Estado_Liquidacao_Gestao': estado_liquidacao,
            'Tipo_Operacao': tp_op,
            'Observacao_Origem': obs_origem,
            'Status_Conciliacao': status_match,
            'Regra_Aplicada': regra_match,
            'ID_OS': detalhes.get('ID_OS', 'N/A'),
            'Numero_NF': detalhes.get('Numero_NF', 'N/A'),
            'ID_Titulo': detalhes.get('ID_Titulo', 'N/A'),
            'Data_Emissao_NF': detalhes.get('Data_Emissao_NF', pd.NaT),
            'Status_Titulo_Truckpag': detalhes.get('Status_Titulo_Truckpag', 'N/A')
        })

    return pd.DataFrame(resultados)


# ==============================================================================
# MÓDULO C: AUDITORIA E TRAVAS DE SEGURANÇA
# ==============================================================================

def aplicar_auditoria_e_travas(df_conciliado, data_corte_param, col_status=None):
    df_audit = df_conciliado.copy()
    dt_corte = pd.to_datetime(data_corte_param)

    # Infração: Emitido a partir de 17/06 e pago via PIX
    cond_infracao_corte = (
            (df_audit['Data_Emissao_NF'] >= dt_corte) &
            (df_audit['Tipo_Operacao'].astype(str).str.upper().str.contains('PIX', na=False)) &
            (df_audit['Status_Conciliacao'] != 'NAO_LOCALIZADO')
    )

    df_audit['Alerta_Infracao_Corte_1706'] = cond_infracao_corte
    df_audit.loc[cond_infracao_corte, 'Status_Conciliacao'] = 'DIVERGENCIA_CORTE_1706'

    # Regra FIFO Refinada: Pagamento efetuado antes da emissão da NF E Status Baixado
    cond_erro_fifo = (
            (df_audit['Data_Pagamento_DGranel'] < df_audit['Data_Emissao_NF']) &
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
# MÓDULO D: RELATÓRIOS E PAYLOAD WHATSAPP
# ==============================================================================

def gerar_payload_whatsapp(df_audit):
    df_pix = df_audit[
        (df_audit['Status_Conciliacao'].isin([
            'CONCILIADO_EXATO', 'ANTECIPACAO_TOTAL_PIX', 'CONCILIADO_POR_VALOR', 'CONCILIADO_OS_MULTISSEGMENTADA'
        ]))
    ]

    if df_pix.empty:
        return "Nenhuma baixa via PIX pendente identificada para envio manual."

    linhas_msg = ["*SOLICITAÇÃO DE BAIXA BANCÁRIA - DGRANEL (PIX)*\n"]
    for idx, row in df_pix.reset_index(drop=True).iterrows():
        dt_str = row['Data_Pagamento_DGranel'].strftime('%d/%m/%Y') if pd.notna(
            row['Data_Pagamento_DGranel']) else 'N/A'
        msg = (
            f"🔹 *Item {idx + 1}*\n"
            f"• *Data Pagto DGranel:* {dt_str}\n"
            f"• *Valor Pago:* R$ {row['Valor_Pago_DGranel']:,.2f}\n"
            f"• *Valor Título:* R$ {row['Valor_Titulo']:,.2f}\n"
            f"• *NF / Doc:* {row['Numero_NF']}\n"
            f"• *Título/Parcela:* {row['ID_Titulo']}\n"
            f"• *OS:* {row['ID_OS']}\n"
            f"• *Estado na Gestão:* {row['Estado_Liquidacao_Gestao']}\n"
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

st.title("🚛 Sistema de Conciliação Inteligente - TruckPag & DGranel")
st.markdown("Cruzamento em cascata (Extratos + Pagamentos PIX vs. Gestão TruckPag).")

# Parâmetros na barra lateral
st.sidebar.header("⚙️ Parâmetros do Sistema")
param_data_inicio = st.sidebar.date_input("Analisar a partir de:", datetime(2026, 8, 3))
param_data_corte = st.sidebar.date_input("Data de Corte (Boleto Obrigatório)", datetime(2026, 6, 17))
param_janela_dias = st.sidebar.slider("Janela Temporal de Busca (Dias)", min_value=1, max_value=30, value=10)
param_tolerancia = st.sidebar.number_input("Tolerância de Valor (R$)", min_value=0.0, max_value=10.0, value=0.05,
                                           step=0.05)

st.sidebar.divider()
st.sidebar.info(
    "Ajuste a tolerância em R$ caso o cliente tenha efetuado pagamentos com pequenas variações de centavos.")

# Carga de Ficheiros com os novos nomes solicitados
st.subheader("📁 Carga dos Arquivos para Conciliação")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**1. Gestão TruckPag (Faturamento) (CSV)**")
    file_fat = st.file_uploader("Upload Gestão TruckPag", type=["csv"])

with col2:
    st.markdown("**2. Extratos Bancários DGranel (PDF/CSV)**")
    files_ext = st.file_uploader("Upload Extratos Bancários", type=["pdf", "csv"], accept_multiple_files=True)

with col3:
    st.markdown("**3. Pagamentos Recebidos via PIX (CSV/PDF)**")
    file_fin = st.file_uploader("Upload Listagem de Pagamentos PIX", type=["csv", "pdf"])

st.divider()

if st.button("⚡ Processar Conciliação e Auditoria", type="primary"):
    if not file_fat or (not files_ext and not file_fin):
        st.error(
            "⚠️ Envie o ficheiro da Gestão TruckPag e pelo menos uma fonte de pagamento (Extrato Bancário ou Pagamentos PIX).")
    else:
        with st.spinner("Processando ETL, cruzamento e validação de datas e valores..."):

            # 1. Gestão TruckPag (Faturamento)
            df_fat = carregar_csv_autodetect(file_fat)
            col_status_fat = encontrar_coluna(df_fat, ['STATUS', 'SITUAÇÃO', 'SITUACAO', 'STATUS_TITULO'])

            # 2. Extratos Bancários DGranel
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

            # 3. Pagamentos Recebidos via PIX
            df_pix_tot = None
            if file_fin:
                if file_fin.name.lower().endswith('.csv'):
                    df_pix_raw = carregar_csv_autodetect(file_fin)
                else:
                    df_pix_raw = extrair_texto_e_tabelas_pdf(file_fin)

                if not df_pix_raw.empty:
                    c_dt_pix = encontrar_coluna(df_pix_raw, ['DATA PAG', 'DATA', 'LANÇAMENTO']) or df_pix_raw.columns[0]
                    c_doc_pix = encontrar_coluna(df_pix_raw, ['TITULO', 'TÍTULO', 'NF', 'DOCUMENTO']) or \
                                df_pix_raw.columns[0]
                    c_val_pix = encontrar_coluna(df_pix_raw, ['VALOR', 'VALOR_PAGO']) or df_pix_raw.columns[1]
                    c_obs_pix = encontrar_coluna(df_pix_raw, ['COMO DEVERIA', 'PAGO VIA', 'STATUS', 'OBS'])

                    lista_pix = []
                    for _, r in df_pix_raw.iterrows():
                        obs = str(r[c_obs_pix]) if c_obs_pix and c_obs_pix in r else "Pagamento PIX"
                        lista_pix.append({
                            'Data_Parsed': parse_data_portugues(r[c_dt_pix]),
                            'Doc_Original': r[c_doc_pix],
                            'Doc_Limpo': limpar_documento(r[c_doc_pix]),
                            'Valor_Pago': limpar_valor_monetario(r[c_val_pix]),
                            'Tipo_Operacao': 'PIX',
                            'Observacao_Origem': f"Listagem PIX: {obs}"
                        })
                    df_pix_tot = pd.DataFrame(lista_pix)

            # Cruzamento entre Listagem PIX e Extrato Bancário
            df_pagamentos_unificados = fundir_pix_com_extrato(
                df_pix=df_pix_tot,
                df_ext=df_ext_tot,
                tolerancia=param_tolerancia
            )

            # Aplicação do Filtro de Data Inicial (ex: a partir de 03/08/2026)
            if not df_pagamentos_unificados.empty:
                dt_inicio_dt = pd.to_datetime(param_data_inicio)
                df_pagamentos_unificados = df_pagamentos_unificados[
                    df_pagamentos_unificados['Data_Parsed'] >= dt_inicio_dt
                    ].copy()

            # Normalização de datas no relatório da Gestão TruckPag
            col_emissao_fat = encontrar_coluna(df_fat, ['EMISSAO', 'EMISSÃO', 'DATA_EMISSAO', 'DATA DE EMISSAO'])
            if col_emissao_fat and col_emissao_fat in df_fat.columns:
                df_fat[col_emissao_fat] = df_fat[col_emissao_fat].apply(parse_data_portugues)

            # Painel Diagnóstico
            with st.expander("🔍 **Inspecionar Leitura dos Dados e Pagamentos Reconciliados (Debug)**", expanded=False):
                d1, d2 = st.columns(2)
                with d1:
                    st.write(f"**Gestão TruckPag (Faturamento):** {len(df_fat)} linhas")
                    st.dataframe(df_fat.head(3), use_container_width=True)
                with d2:
                    st.write(
                        f"**Pagamentos Unificados para Conciliação (A partir de {param_data_inicio.strftime('%d/%m/%Y')}):** {len(df_pagamentos_unificados)} linhas")
                    st.dataframe(df_pagamentos_unificados.head(5), use_container_width=True)

            # Motor em Cascata (Waterfall)
            df_conciliado = executar_waterfall_matching(
                df_pagamentos=df_pagamentos_unificados,
                df_faturamento=df_fat,
                janela_dias=param_janela_dias,
                tolerancia=param_tolerancia
            )

            # Auditoria e Regras de Segurança
            df_resultado_final = aplicar_auditoria_e_travas(
                df_conciliado=df_conciliado,
                data_corte_param=param_data_corte,
                col_status=col_status_fat
            )

            st.success(
                f"✅ Processamento e auditoria concluídos com sucesso! {len(df_resultado_final)} transações analisadas.")

            # Apresentação do Painel de Resultados
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
                st.write("### Pagamentos sem correspondência encontrada na Gestão TruckPag")
                st.dataframe(df_resultado_final[df_resultado_final['Status_Conciliacao'] == 'NAO_LOCALIZADO'],
                             use_container_width=True)

            with tab6:
                st.write("### Payload Formatado para Envio ao Grupo do WhatsApp")
                st.code(gerar_payload_whatsapp(df_resultado_final), language="text")

            excel_bytes = gerar_excel_download(df_resultado_final)
            st.download_button(
                label="📥 Baixar Relatório Completo em Excel (Multi-Abas)",
                data=excel_bytes,
                file_name=f"Conciliacao_TruckPag_DGranel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )