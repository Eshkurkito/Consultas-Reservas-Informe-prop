
"""
owner_report_module_v2.py
-------------------------
Módulo plug-and-play para generar un Informe de Propietario (PDF) en Streamlit
SIN depender de funciones de la app anfitriona (ni `period_inputs` ni `compute_kpis`).

- Calcula KPIs básicos a partir de un dataframe de reservas con columnas comunes.
- Usa fpdf2 + DejaVuSans (Unicode) si está disponible; si no, hace fallback seguro.
- Color corporativo: #163e64

Requisitos en requirements.txt:
    fpdf2==2.7.9

Coloca la fuente Unicode (opcional pero recomendado) en alguno de estos paths:
    ./assets/fonts/DejaVuSans.ttf
    ./fonts/DejaVuSans.ttf
    ./DejaVuSans.ttf
    /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
"""

from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, List

import os
import numpy as np
import pandas as pd
import streamlit as st

# ===============================
# --------- UTILIDADES ----------
# ===============================

BRAND_RGB = (0x16, 0x3e, 0x64)   # #163e64
DARK_RGB  = (51, 51, 51)

_DEJAVU_CANDIDATES = [
    "./assets/fonts/DejaVuSans.ttf",
    "./fonts/DejaVuSans.ttf",
    "./DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def _find_dejavu() -> Optional[str]:
    for p in _DEJAVU_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

def _safe_text(s: str, unicode_ok: bool) -> str:
    if unicode_ok:
        return s
    return (
        s.replace("•", "-")
         .replace("–", "-")
         .replace("—", "-")
         .replace("€", "EUR")
         .replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
         .replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
         .replace("ñ", "n").replace("Ñ", "N")
    )

def _find_col(df: pd.DataFrame, candidates: List[str], contains: bool=False) -> Optional[str]:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if contains:
            for c in cols:
                if cand.lower() in c.lower():
                    return c
        else:
            if cand.lower() in low:
                return low[cand.lower()]
    return None

def _parse_date_series(s: pd.Series) -> pd.Series:
    # Acepta datetime, date, string
    s2 = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False).dt.tz_localize(None)
    # convertir a date sin hora para cálculo de noches (entrada inclusive, salida exclusiva)
    return s2.dt.date

def _kpis_fallback(
    df_all: pd.DataFrame,
    cutoff: date,
    period_start: date,
    period_end: date,
    inventory_override: Optional[int]=None,
    filter_props: Optional[List[str]]=None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Calcula KPIs básicos a partir de reservas (nivel reserva). Asume columnas típicas:
      - Alojamiento (u otras variantes)
      - Fecha entrada / Fecha salida (o check-in/out)
      - Fecha alta (fecha de creación), opcional
      - Ingresos (o Total/Importe/Revenue) opcional
      - Precio por noche (si no hay ingresos)
    Devuelve:
      - by_prop: KPIs por alojamiento
      - tot: dict con KPIs totales
    """
    df = df_all.copy()

    # Mapeo de columnas
    aloj = _find_col(df, ["Alojamiento", "Unidad", "Apartamento", "Propiedad", "Listing", "Unidad/Alojamiento", "Nombre alojamiento"])
    fin  = _find_col(df, ["Fecha salida", "Salida", "Check-out", "Fecha de salida", "Departure"], contains=True) or _find_col(df, ["Salida", "Check-out"])
    ini  = _find_col(df, ["Fecha entrada", "Entrada", "Check-in", "Fecha de entrada", "Arrival"], contains=True) or _find_col(df, ["Entrada", "Check-in"])
    alta = _find_col(df, ["Fecha alta", "Creación", "Fecha reserva", "Booking Date", "Fecha de creación"], contains=True)

    rev  = None
    for key in ["ingres", "revenue", "importe", "total", "aloj"]:
        rev = _find_col(df, [key], contains=True)
        if rev and np.issubdtype(pd.to_datetime(df[rev], errors="ignore").dtype, np.datetime64):
            # Evitar confundir totales con fechas; si es fecha, ignora
            rev = None
        if rev:
            # debe ser numérico
            try:
                df[rev] = pd.to_numeric(df[rev], errors="coerce")
            except Exception:
                rev = None
        if rev: break

    price_col = None
    for key in ["precio", "tarifa", "rate", "adr"]:
        price_col = _find_col(df, [key], contains=True)
        if price_col:
            try:
                df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
            except Exception:
                price_col = None
        if price_col: break

    # Validaciones mínimas
    if not aloj:
        raise ValueError("No se encontró columna de alojamiento (ej. 'Alojamiento', 'Unidad', 'Apartamento', 'Listing').")
    if not ini or not fin:
        raise ValueError("Faltan columnas de fechas de entrada/salida (ej. 'Fecha entrada' y 'Fecha salida').")

    # Filtros por cutoff (si hay fecha alta)
    if alta and alta in df.columns:
        df = df[pd.to_datetime(df[alta], errors="coerce") <= pd.to_datetime(cutoff)]

    # Filtro por propiedades
    if filter_props:
        df = df[df[aloj].isin(filter_props)]

    # Parse fechas
    df["_in"]  = _parse_date_series(df[ini])
    df["_out"] = _parse_date_series(df[fin])

    # Eliminar filas inválidas
    df = df.dropna(subset=["_in", "_out"])
    df = df[df["_out"] > df["_in"]]

    # Inventario
    inv = df[aloj].nunique() if aloj in df.columns else 0
    if inventory_override and inventory_override > 0:
        inv = int(inventory_override)

    # Noches disponibles
    days = (period_end - period_start).days + 1
    nights_available = max(inv * days, 0)

    # Calcular noches vendidas y revenue solapado
    nights_sold = 0
    revenue = 0.0
    # También por propiedad
    agg = {}

    # Precompute period_end exclusive for night calc
    pe_exclusive = period_end + timedelta(days=1)

    for _, row in df.iterrows():
        r_in, r_out = row["_in"], row["_out"]
        # Overlap
        start = max(r_in, period_start)
        end   = min(r_out, pe_exclusive)
        n = (end - start).days
        if n <= 0:
            continue

        # revenue prorrateado
        rev_row = 0.0
        if rev and pd.notna(row.get(rev, np.nan)):
            total_n = (r_out - r_in).days
            total_n = total_n if total_n > 0 else n
            frac = n / total_n
            try:
                rev_row = float(row.get(rev, 0.0)) * frac
            except Exception:
                rev_row = 0.0
        elif price_col and pd.notna(row.get(price_col, np.nan)):
            try:
                rev_row = float(row.get(price_col, 0.0)) * n
            except Exception:
                rev_row = 0.0

        nights_sold += n
        revenue += rev_row

        k = row[aloj]
        if k not in agg:
            agg[k] = {"nights": 0, "rev": 0.0}
        agg[k]["nights"] += n
        agg[k]["rev"]    += rev_row

    adr   = (revenue / nights_sold) if nights_sold > 0 else 0.0
    revpar= (revenue / nights_available) if nights_available > 0 else 0.0
    occ   = (nights_sold / nights_available * 100.0) if nights_available > 0 else 0.0

    by_prop = (
        pd.DataFrame([
            { "Alojamiento": k,
              "Noches ocupadas": v["nights"],
              "Ingresos (€)": v["rev"],
              "ADR (€)": (v["rev"]/v["nights"]) if v["nights"]>0 else 0.0
            }
            for k, v in agg.items()
        ])
        .sort_values("Ingresos (€)", ascending=False)
        if agg else pd.DataFrame(columns=["Alojamiento","Noches ocupadas","Ingresos (€)","ADR (€)"])
    )

    tot = {
        "noches_ocupadas": nights_sold,
        "noches_disponibles": nights_available,
        "ocupacion_pct": occ,
        "ingresos": revenue,
        "adr": adr,
        "revpar": revpar,
    }

    return by_prop, tot

# ===============================
# --------- MODO UI -------------
# ===============================

def render_owner_report_mode():
    st.header("Informe de Propietario (PDF)")
    raw = st.session_state.get("raw_df")
    if raw is None or raw.empty:
        st.warning("Carga un archivo de reservas para generar el informe.")
        st.stop()

    with st.sidebar:
        st.header("Parámetros del informe")
        cutoff_rep = st.date_input("Fecha de corte", value=date.today(), key="cutoff_owner")
        default_start = date(date.today().year, date.today().month, 1)
        default_end   = (pd.Timestamp.today().to_period("M").end_time).date()
        p_start = st.date_input("Inicio del periodo", value=default_start, key="pstart_owner")
        p_end   = st.date_input("Fin del periodo", value=default_end, key="pend_owner")

        aloj_col = _find_col(raw, ["Alojamiento", "Unidad", "Apartamento", "Propiedad", "Listing", "Nombre alojamiento"])
        prop_options = sorted(raw[aloj_col].dropna().unique()) if aloj_col else []
        props_rep = st.multiselect("Filtrar alojamientos (opcional)", options=prop_options, default=[])

        inv_rep = st.number_input("Sobrescribir inventario (opcional)", min_value=0, value=0, step=1, key="inv_owner")

        propietario = st.text_input("Nombre del propietario", value="Propietario")
        propiedad   = st.text_input("Nombre del apartamento/grupo", value="Portafolio")
        mes_text    = st.text_input("Periodo (texto portada)", value=pd.to_datetime(p_start).strftime("%B %Y").capitalize())

    # KPIs con fallback interno
    by_prop, tot = _kpis_fallback(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_rep).date(),
        period_start=p_start,
        period_end=p_end,
        inventory_override=int(inv_rep) if inv_rep > 0 else None,
        filter_props=props_rep if props_rep else None,
    )

    # Vista previa rápida
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Noches ocupadas", f"{int(tot['noches_ocupadas']):,}".replace(",", "."))
    c2.metric("Noches disponibles", f"{int(tot['noches_disponibles']):,}".replace(",", "."))
    c3.metric("Ocupación", f"{tot['ocupacion_pct']:.2f}%")
    c4.metric("Ingresos (€)", f"{tot['ingresos']:.2f}")
    c5.metric("ADR (€)", f"{tot['adr']:.2f}")
    c6.metric("RevPAR (€)", f"{tot['revpar']:.2f}")
    st.divider()

    # -------- Generación PDF (fpdf2 + Unicode si disponible) --------
    try:
        from fpdf import FPDF

        font_path = _find_dejavu()
        unicode_ok = font_path is not None

        class OwnerReport(FPDF):
            def header(self):
                self.set_fill_color(*BRAND_RGB)
                self.rect(x=10, y=10, w=5, h=277, style="F")

            def footer(self):
                self.set_y(-20)
                self.set_fill_color(*BRAND_RGB)
                self.rect(x=15, y=self.get_y()+8, w=180, h=1.2, style="F")
                self.set_y(-15)
                self.set_text_color(*DARK_RGB)
                if unicode_ok:
                    self.set_font("DejaVu", size=8)
                else:
                    self.set_font("Helvetica", size=8)
                self.cell(0, 10, _safe_text("FLORIT FLATS • Informe de Propietario", unicode_ok), 0, 0, "L")
                self.cell(0, 10, f"Página {self.page_no()}", 0, 0, "R")

        pdf = OwnerReport(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)

        # Fuentes
        if unicode_ok:
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.add_font("DejaVu", "B", font_path, uni=True)
            title_font = ("DejaVu", "B", 22)
            h2_font    = ("DejaVu", "B", 16)
            body_font  = ("DejaVu", "", 11)
            val_font   = ("DejaVu", "B", 14)
            tag_font   = ("DejaVu", "", 9)
        else:
            title_font = ("Helvetica", "B", 22)
            h2_font    = ("Helvetica", "B", 16)
            body_font  = ("Helvetica", "", 11)
            val_font   = ("Helvetica", "B", 14)
            tag_font   = ("Helvetica", "", 9)

        # -------- Portada --------
        pdf.add_page()
        pdf.set_text_color(*BRAND_RGB)
        pdf.set_font(*title_font)
        pdf.set_xy(25, 30)
        pdf.cell(0, 10, _safe_text("Informe de Propietario", unicode_ok), ln=1)

        pdf.set_text_color(*DARK_RGB)
        pdf.set_font(*body_font)
        pdf.set_xy(25, 45)
        pdf.cell(0, 8, _safe_text(f"Periodo: {mes_text}", unicode_ok), ln=1)
        pdf.cell(0, 8, _safe_text(f"Propietario: {propietario}", unicode_ok), ln=1)
        pdf.cell(0, 8, _safe_text(f"Propiedad: {propiedad}", unicode_ok), ln=1)

        # Caja hero
        pdf.set_xy(25, 80)
        pdf.set_draw_color(220,220,220)
        pdf.set_fill_color(245,245,245)
        pdf.cell(160, 60, "", border=1, fill=True, ln=1)
        pdf.set_xy(25, 110)
        pdf.cell(160, 8, _safe_text("Espacio para foto (Canva) o gráfico (PNG).", unicode_ok), ln=1, align="C")

        # -------- Resumen & KPIs --------
        pdf.add_page()
        pdf.set_text_color(*BRAND_RGB)
        pdf.set_font(*h2_font)
        pdf.set_xy(25, 20)
        pdf.cell(0, 8, _safe_text("Resumen ejecutivo", unicode_ok), ln=1)

        pdf.set_text_color(*DARK_RGB)
        pdf.set_font(*body_font)
        resumen = (
            f"En {mes_text}, la ocupación fue del {tot['ocupacion_pct']:.2f}%, "
            f"el ADR se situó en {tot['adr']:.2f} € y el RevPAR en {tot['revpar']:.2f} €, "
            f"generando unos ingresos totales de {tot['ingresos']:.2f} €.\n"
            "Principales drivers: [canales/eventos/segmentos]."
        )
        pdf.set_xy(25, 32)
        pdf.multi_cell(160, 6, _safe_text(resumen, unicode_ok))

        # Tarjetas KPI (2x3)
        def kpi_card(x, y, label, value):
            pdf.set_draw_color(200,200,200)
            pdf.set_fill_color(255,255,255)
            pdf.rect(x, y, 50, 28, style="DF")
            pdf.set_fill_color(*BRAND_RGB)
            pdf.rect(x, y, 50, 5, style="F")
            pdf.set_xy(x, y+8)
            pdf.set_text_color(*BRAND_RGB)
            pdf.set_font(*val_font)
            pdf.cell(50, 8, _safe_text(value, unicode_ok), align="C")
            pdf.set_xy(x, y+18)
            pdf.set_text_color(*DARK_RGB)
            pdf.set_font(*tag_font)
            pdf.cell(50, 6, _safe_text(label, unicode_ok), align="C")

        cards = [
            ("Ocupación (%)", f"{tot['ocupacion_pct']:.2f}%"),
            ("ADR (€)",       f"{tot['adr']:.2f}"),
            ("RevPAR (€)",    f"{tot['revpar']:.2f}"),
            ("Ingresos (€)",  f"{tot['ingresos']:.2f}"),
            ("Noches ocupadas", f"{int(tot['noches_ocupadas'])}"),
            ("Noches disponibles", f"{int(tot['noches_disponibles'])}"),
        ]
        x0, y0 = 25, 70
        gap_x, gap_y = 10, 10
        for i, (lab, val) in enumerate(cards):
            col, row = i % 3, i // 3
            x = x0 + col*(50 + gap_x)
            y = y0 + row*(28 + gap_y)
            kpi_card(x, y, lab, val)

        # Output
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        st.download_button(
            "📄 Descargar Informe PDF",
            data=pdf_bytes,
            file_name=f"Informe_Propietario_{mes_text.replace(' ','_')}.pdf",
            mime="application/pdf",
            type="primary"
        )
        st.success("Informe generado (módulo independiente). Próximo paso: insertar gráficos/heatmap en PNG.")
    except Exception as e:
        st.error(f"No se pudo generar el PDF ({e}).")
        st.info("Comprueba que 'fpdf2' está instalado y que la fuente DejaVuSans.ttf está disponible (opcional).")
