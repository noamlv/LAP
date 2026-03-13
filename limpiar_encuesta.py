#!/usr/bin/env python3
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
IN_FILE = BASE_DIR / "Encuesta" / "Encuesta a la comunidad venezolana en Cusco (respuestas).xlsx"
OUT_DIR = BASE_DIR / "datasets_finales"
LIMPIOS_DIR = OUT_DIR / "01_limpios"
DICC_PATH = OUT_DIR / "02_diccionario" / "diccionario_unificado.csv"

OUT_DATASET = LIMPIOS_DIR / "encuesta_comunidad_venezolana.csv"


def norm_name(s: str) -> str:
    s = str(s).strip().replace("\n", " ")
    s = " ".join(s.split())
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def squish_text(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\u00a0", " ")
    s = " ".join(s.split())
    return s if s else np.nan


def strip_choice_prefix(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"^\d+\s*[\.)-]?\s*", "", s)
    s = s.replace("→", " ")
    s = " ".join(s.split())
    return s if s else np.nan


def to_yes_no(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = re.sub(r"^\d+\s*[\.)-]?\s*", "", s)
    s = s.replace("sí", "si")
    if "no recuerda" in s:
        return "No recuerda"
    if s.startswith("si"):
        return "Sí"
    if s.startswith("no"):
        return "No"
    return np.nan


def extract_year(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r"(19\d{2}|20\d{2})", s)
    return int(m.group(1)) if m else np.nan


def parse_datetime_col(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if dt.notna().mean() < 0.2:
        # Para entradas como "mes/año" o texto mezclado, intentamos parseo flexible extra.
        dt = pd.to_datetime(s.astype(str).str.replace(r"[^0-9/\-]", "", regex=True), errors="coerce", dayfirst=True)
    return dt


def parse_hours_midpoint(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x)
    nums = [int(n) for n in re.findall(r"\d+", s)]
    if not nums:
        return np.nan
    if len(nums) == 1:
        return float(nums[0])
    return float((nums[0] + nums[1]) / 2)


def parse_days(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r"\d+", s)
    return float(m.group(0)) if m else np.nan


def parse_money(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x)
    nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", s)]
    if not nums:
        return np.nan
    if len(nums) == 1:
        return float(nums[0])
    return float((nums[0] + nums[1]) / 2)


def parse_ingreso_midpoint(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = strip_choice_prefix(x)
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = s.replace("–", "-")
    if "menos" in s.lower() and "600" in s:
        return 500.0
    if "mas de" in s.lower() or "más de" in s.lower():
        nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", s)]
        return float(nums[-1]) if nums else 3000.0
    nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", s)]
    if len(nums) >= 2:
        return float((nums[0] + nums[1]) / 2)
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def main() -> None:
    LIMPIOS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(IN_FILE)
    raw.columns = [norm_name(c) for c in raw.columns]

    # Limpieza básica de texto
    for c in raw.columns:
        if raw[c].dtype == object:
            raw[c] = raw[c].map(squish_text)

    # Renombre a nombres analíticos cortos
    rename_map = {
        "cual_es_su_estado_civil": "estado_civil",
        "en_que_fecha_llego_al_peru_indique_mes_y_ano": "fecha_llegada_peru_raw",
        "en_que_fecha_llego_al_cusco_indique_mes_y_ano": "fecha_llegada_cusco_raw",
        "en_que_distrito_del_cusco_reside_usted": "distrito_residencia",
        "el_sistema_de_seguridad_d_salud_al_cual_usted_es_afiliado_actualmente_es": "seguro_salud",
        "padece_alguna_enfermedad_o_malestar_cronico": "enfermedad_cronica",
        "si_padece_alguna_enfermedad_o_malestar_cronico_mencione_cual": "enfermedad_cronica_detalle",
        "si_padece_alguna_enfermedad_o_malestar_cronico_recibe_tratamiento": "tratamiento_cronico",
        "tiene_hijos_o_hijas": "tiene_hijos",
        "cual_es_el_ultimo_nivel_de_estudio_que_aprobo_en_venezuela": "nivel_estudio",
        "si_cuenta_con_formacion_superior_indique_el_nombre_de_la_carrera_o_especialidad_universitaria_o_tecnica_que_cursa_o_ha_cursado": "carrera_formacion_superior",
        "de_contar_con_estudios_superiores_concluidos_en_venezuela_su_titulo_fue_homologado_en_peru": "titulo_homologado_peru",
        "por_que_no_ha_homologado_su_titulo": "motivo_no_homologacion",
        "el_ano_pasado_estuvo_inscrito_matriculado_en_algun_centro_o_porgrama_de_educacion_superior": "inscrito_sup_anio_pasado",
        "este_ano_esta_inscrito_matriculado_en_algun_centro_o_porgrama_de_educacion_superior": "inscrito_sup_este_anio",
        "actualmente_asiste_a_algun_centro_o_programa_de_educacion_basica_superior": "asiste_educ_basica_superior",
        "en_venezuela_tenia_usted_trabajo_antes_de_iniciar_su_viaje": "trabajo_previo_venezuela",
        "cual_es_la_ocupacion_principal_que_desempena_actualmente": "ocupacion_actual",
        "cual_es_su_situacion_laboral_actual": "situacion_laboral",
        "cual_es_su_tipo_de_empleo_solo_si_actualmente_trabaja": "tipo_empleo",
        "su_trabajo_actual_se_relaciona_con_su_formacion_academica_o_experiencia_previa": "trabajo_relacion_formacion",
        "cuantas_horas_trabaja_normalmente_en_su_ocupacion_principal": "horas_trabajo_cat",
        "en_promedio_cuantos_dias_trabaja_por_semana_anote_el_numero_de_dias": "dias_trabajo_semana_raw",
        "cual_es_su_ingreso_mensual_aproximado_por_su_trabajo_principal_solo_en_dinero": "ingreso_categoria",
        "en_los_ultimos_3_meses_envio_remesas_dinero_a_otro_pais": "envia_remesas",
        "si_si_a_que_pais_envio_remesas": "pais_remesas",
        "con_que_frecuencia_envio_remesas_en_ese_periodo": "frecuencia_remesas",
        "monto_aproximado_que_envio_el_ultimo_mes_en_soles": "monto_remesa_ultimo_mes_raw",
        "como_consiguio_su_trabajo_actual_puede_marcar_mas_de_una_opcion": "canal_conseguir_trabajo",
        "hace_cuanto_tiempo_trabaja_en_este_empleo_u_ocupacion_actual": "tiempo_en_empleo_actual",
        "considera_que_su_situacion_economica_actual_versus_la_que_enfrentaba_en_el_momento_de_su_llegada_es": "situacion_economica_vs_llegada",
        "ha_sentido_que_tiene_menos_oportunidades_laborales_o_recibe_menor_salario_que_personas_peruanas_con_igual_calificacion_solo_por_ser_venezolano_a": "percibe_discriminacion_laboral",
        "que_tan_satisfecho_a_se_siente_con_su_trabajo_actual": "satisfaccion_laboral",
        "desde_su_llegada_a_cusco_ha_recibido_apoyo_de_programas_de_cooperacion_internacional_ongs": "recibio_apoyo_ong",
        "si_respondio_si_que_organizaciones_que_le_brindaron_apoyo_marque_todas_las_opciones": "organizaciones_apoyo_ong",
        "en_que_ano_recibio_apoyo_de_programas_de_asistencia_humanitaria_de_una_ong_puede_marcar_mas_de_una_opcion": "anios_apoyo_ong",
        "que_tipo_de_asistencia_recibio_puede_marcar_mas_de_una_opcion": "tipo_asistencia_ong",
        "con_que_frecuencia_recibio_la_asistencia": "frecuencia_asistencia",
        "pudo_acceder_a_atencion_en_salud_gracias_a_la_asistencia_humanitaria_recibida": "acceso_salud_por_asistencia",
        "si_respondio_si_percibio_mejoras_en_su_salud_o_la_de_su_familia_tras_recibir_apoyo": "mejora_salud_tras_apoyo",
        "algun_miembro_de_su_hogar_recibio_apoyo_educativo_matricula_utiles_becas_talleres_en_2024_por_parte_de_alguna_ong": "apoyo_educativo_hogar_2024",
        "si_respondio_si_este_apoyo_ayudo_a_que_sus_hijos_hijas_continuen_en_la_escuela": "apoyo_ayudo_continuidad_escolar",
        "recibio_algun_apoyo_relacionado_con_medios_de_vida_en_el_ano_2024_puede_marcar_mas_de_una_opcion": "apoyo_medios_vida_2024",
        "el_apoyo_mejoro_sus_ingresos_o_estabilidad_economica": "mejora_ingresos_por_apoyo",
        "actualmente_sigue_usando_los_conocimientos_herramientas_recibidas": "usa_herramientas_apoyo",
        "si_marco_la_opcion_3_porque_no_aplica_los_conocimiento_herramientas": "motivo_no_uso_herramientas",
        "de_que_organismos_del_estado_peruano_recibio_asistencia_o_apoyo_puede_marcar_mas_de_una_opcion": "organismos_estado_apoyo",
        "en_que_dia_mes_ano_nacio": "fecha_nacimiento_raw",
        "cual_es_su_estado_de_procedencia": "estado_procedencia",
        "en_que_mes_y_ano_ingreso_al_peru_por_ultima_vez": "fecha_ultimo_ingreso_peru_raw",
        "por_que_ciudad_ingreso_a_peru_la_ultima_vez": "ciudad_ultimo_ingreso_peru",
        "que_documentos_de_identidad_de_su_pais_tiene_con_usted": "documentos_identidad",
        "actualmente_que_tipo_de_permiso_migratorio_tiene_para_estar_en_peru": "permiso_migratorio_actual",
        "responda_la_pregunta_si_no_cuenta_con_permiso_migratorio_cual_es_la_razon_principal_por_la_que_no_cuenta_con_permiso_migratorio_para_estar_pregunta": "motivo_sin_permiso_migratorio",
        "usted_ha_solicitado_refugio": "solicito_refugio",
        "donde_planea_residir_en_los_proximos_anos": "residencia_proximos_anios",
    }

    df = raw.rename(columns=rename_map).copy()

    # Campos base
    df.insert(0, "fuente", "ENCUESTA")
    df.insert(1, "subfuente", "Encuesta comunidad venezolana en Cusco")
    df.insert(2, "id_encuesta", range(1, len(df) + 1))

    # Limpieza de categorías codificadas
    choice_cols = [
        "sexo", "estado_civil", "distrito_residencia", "seguro_salud", "enfermedad_cronica",
        "tratamiento_cronico", "tiene_hijos", "nivel_estudio", "titulo_homologado_peru",
        "inscrito_sup_anio_pasado", "inscrito_sup_este_anio", "asiste_educ_basica_superior",
        "trabajo_previo_venezuela", "situacion_laboral", "tipo_empleo", "trabajo_relacion_formacion",
        "ingreso_categoria", "satisfaccion_laboral", "recibio_apoyo_ong", "frecuencia_asistencia",
        "acceso_salud_por_asistencia", "mejora_salud_tras_apoyo", "apoyo_educativo_hogar_2024",
        "apoyo_ayudo_continuidad_escolar", "mejora_ingresos_por_apoyo", "usa_herramientas_apoyo",
        "permiso_migratorio_actual", "solicito_refugio", "residencia_proximos_anios"
    ]
    for c in choice_cols:
        if c in df.columns:
            df[c] = df[c].map(strip_choice_prefix)

    # Ajustes de consistencia para sí/no
    yn_cols = [
        "enfermedad_cronica", "trabajo_relacion_formacion", "envia_remesas", "recibio_apoyo_ong",
        "acceso_salud_por_asistencia", "apoyo_educativo_hogar_2024", "apoyo_ayudo_continuidad_escolar"
    ]
    for c in yn_cols:
        if c in df.columns:
            df[c] = df[c].map(to_yes_no)

    # Variables numéricas y de fecha
    df["edad"] = pd.to_numeric(df.get("edad"), errors="coerce")
    df["horas_trabajo_aprox"] = df.get("horas_trabajo_cat").map(parse_hours_midpoint)
    df["dias_trabajo_semana"] = df.get("dias_trabajo_semana_raw").map(parse_days)
    df["ingreso_mensual_aprox_soles"] = df.get("ingreso_categoria").map(parse_ingreso_midpoint)
    df["monto_remesa_ultimo_mes_soles"] = df.get("monto_remesa_ultimo_mes_raw").map(parse_money)

    df["marca_temporal"] = parse_datetime_col(df.get("marca_temporal"))
    df["fecha_llegada_peru"] = parse_datetime_col(df.get("fecha_llegada_peru_raw"))
    df["fecha_llegada_cusco"] = parse_datetime_col(df.get("fecha_llegada_cusco_raw"))
    df["fecha_nacimiento"] = parse_datetime_col(df.get("fecha_nacimiento_raw"))
    df["fecha_ultimo_ingreso_peru"] = parse_datetime_col(df.get("fecha_ultimo_ingreso_peru_raw"))

    df["anio_llegada_peru"] = df["fecha_llegada_peru_raw"].map(extract_year)
    df["anio_llegada_cusco"] = df["fecha_llegada_cusco_raw"].map(extract_year)
    df["anio_nacimiento"] = df["fecha_nacimiento_raw"].map(extract_year)
    df["anio_ultimo_ingreso_peru"] = df["fecha_ultimo_ingreso_peru_raw"].map(extract_year)

    # Correcciones específicas
    if "sexo" in df.columns:
        df["sexo"] = df["sexo"].replace({"Hombre": "Hombre", "Mujer": "Mujer"})

    # Columna con ruido (mezcla de años en pregunta de trabajo previo)
    if "trabajo_previo_venezuela" in df.columns:
        trabajo_prev_raw = df["trabajo_previo_venezuela"].copy()
        df["trabajo_previo_venezuela"] = trabajo_prev_raw.map(to_yes_no)
        df["trabajo_previo_venezuela_detalle_raw"] = trabajo_prev_raw

    # Derivadas inferenciales
    df["empleo_informal_bin"] = np.where(
        df.get("tipo_empleo", "").astype(str).str.contains("informal", case=False, na=False), 1,
        np.where(df.get("tipo_empleo").notna(), 0, np.nan)
    )
    df["remesas_bin"] = np.where(df.get("envia_remesas") == "Sí", 1, np.where(df.get("envia_remesas") == "No", 0, np.nan))
    df["apoyo_ong_bin"] = np.where(df.get("recibio_apoyo_ong") == "Sí", 1, np.where(df.get("recibio_apoyo_ong") == "No", 0, np.nan))
    sat = df.get("satisfaccion_laboral")
    sat_alta = sat.isin(["Satisfecho/a", "Muy satisfecho/a"])
    sat_valida = sat.notna()
    df["satisfaccion_alta_bin"] = np.where(sat_alta, 1, np.where(sat_valida, 0, np.nan))
    df["permanencia_cusco_bin"] = np.where(
        df.get("residencia_proximos_anios", "").astype(str).str.contains("cusco", case=False, na=False),
        1,
        np.where(df.get("residencia_proximos_anios").notna(), 0, np.nan)
    )

    # Selección final de columnas
    keep_cols = [
        "fuente", "subfuente", "id_encuesta", "marca_temporal", "sexo", "edad", "estado_civil",
        "fecha_llegada_peru", "anio_llegada_peru", "fecha_llegada_cusco", "anio_llegada_cusco",
        "distrito_residencia", "seguro_salud", "enfermedad_cronica", "enfermedad_cronica_detalle",
        "tratamiento_cronico", "tiene_hijos", "nivel_estudio", "carrera_formacion_superior",
        "titulo_homologado_peru", "motivo_no_homologacion", "inscrito_sup_anio_pasado",
        "inscrito_sup_este_anio", "asiste_educ_basica_superior", "trabajo_previo_venezuela",
        "trabajo_previo_venezuela_detalle_raw", "ocupacion_actual", "situacion_laboral", "tipo_empleo",
        "trabajo_relacion_formacion", "horas_trabajo_cat", "horas_trabajo_aprox", "dias_trabajo_semana",
        "ingreso_categoria", "ingreso_mensual_aprox_soles", "envia_remesas", "pais_remesas",
        "frecuencia_remesas", "monto_remesa_ultimo_mes_soles", "canal_conseguir_trabajo",
        "tiempo_en_empleo_actual", "situacion_economica_vs_llegada", "percibe_discriminacion_laboral",
        "satisfaccion_laboral", "recibio_apoyo_ong", "organizaciones_apoyo_ong", "anios_apoyo_ong",
        "tipo_asistencia_ong", "frecuencia_asistencia", "acceso_salud_por_asistencia",
        "mejora_salud_tras_apoyo", "apoyo_educativo_hogar_2024", "apoyo_ayudo_continuidad_escolar",
        "apoyo_medios_vida_2024", "mejora_ingresos_por_apoyo", "usa_herramientas_apoyo",
        "motivo_no_uso_herramientas", "organismos_estado_apoyo", "fecha_nacimiento", "anio_nacimiento",
        "estado_procedencia", "fecha_ultimo_ingreso_peru", "anio_ultimo_ingreso_peru",
        "ciudad_ultimo_ingreso_peru", "documentos_identidad", "permiso_migratorio_actual",
        "motivo_sin_permiso_migratorio", "solicito_refugio", "residencia_proximos_anios",
        "empleo_informal_bin", "remesas_bin", "apoyo_ong_bin", "satisfaccion_alta_bin", "permanencia_cusco_bin"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    # Guardar dataset limpio
    out.to_csv(OUT_DATASET, index=False, encoding="utf-8")

    # Actualizar diccionario unificado
    desc_map = {
        "id_encuesta": "Identificador secuencial de registro de encuesta.",
        "marca_temporal": "Fecha y hora de registro de la respuesta.",
        "sexo": "Sexo declarado por la persona encuestada.",
        "edad": "Edad de la persona encuestada en años.",
        "estado_civil": "Estado civil reportado.",
        "anio_llegada_peru": "Año de llegada al Perú.",
        "anio_llegada_cusco": "Año de llegada al Cusco.",
        "distrito_residencia": "Distrito de residencia actual en Cusco.",
        "seguro_salud": "Tipo de afiliación actual al sistema de salud.",
        "nivel_estudio": "Último nivel educativo aprobado en Venezuela.",
        "situacion_laboral": "Situación laboral actual de la persona encuestada.",
        "tipo_empleo": "Tipo de empleo actual (formal/informal/independiente).",
        "horas_trabajo_aprox": "Horas aproximadas trabajadas por día en ocupación principal.",
        "dias_trabajo_semana": "Número de días trabajados por semana.",
        "ingreso_categoria": "Tramo categórico de ingreso mensual laboral.",
        "ingreso_mensual_aprox_soles": "Ingreso laboral mensual aproximado en soles (estimado por tramo).",
        "envia_remesas": "Indica si envía remesas al exterior.",
        "frecuencia_remesas": "Frecuencia declarada de envío de remesas.",
        "monto_remesa_ultimo_mes_soles": "Monto aproximado enviado en el último mes (soles).",
        "satisfaccion_laboral": "Nivel de satisfacción con trabajo actual.",
        "recibio_apoyo_ong": "Indica si recibió apoyo de cooperación internacional (ONG).",
        "tipo_asistencia_ong": "Tipos de asistencia humanitaria recibida.",
        "frecuencia_asistencia": "Frecuencia reportada de la asistencia recibida.",
        "acceso_salud_por_asistencia": "Indica si pudo acceder a salud gracias a asistencia humanitaria.",
        "apoyo_educativo_hogar_2024": "Indica si algún miembro del hogar recibió apoyo educativo en 2024.",
        "apoyo_medios_vida_2024": "Indica si recibió apoyo de medios de vida en 2024.",
        "mejora_ingresos_por_apoyo": "Percepción de mejora de ingresos o estabilidad por el apoyo.",
        "permiso_migratorio_actual": "Tipo de permiso migratorio actual para permanencia en Perú.",
        "solicito_refugio": "Situación de solicitud o reconocimiento de refugio.",
        "residencia_proximos_anios": "Lugar donde planea residir en los próximos años.",
        "empleo_informal_bin": "Indicador binario de empleo informal actual (1=Sí).",
        "remesas_bin": "Indicador binario de envío de remesas (1=Sí).",
        "apoyo_ong_bin": "Indicador binario de recepción de apoyo ONG (1=Sí).",
        "satisfaccion_alta_bin": "Indicador binario de satisfacción laboral alta (1=Satisfecho/Muy satisfecho).",
        "permanencia_cusco_bin": "Indicador binario de plan de permanencia en la provincia del Cusco (1=Sí).",
    }

    tipo_map = {
        "int64": "Int64",
        "float64": "Float64",
        "datetime64[ns]": "datetime",
        "object": "string",
        "bool": "boolean",
    }

    dicc_rows = []
    for c in out.columns:
        t = str(out[c].dtype)
        tipo = tipo_map.get(t, "string")
        desc = desc_map.get(c, f"Variable de encuesta: {c.replace('_', ' ')}.")
        dicc_rows.append(
            {
                "variable_estandar": c,
                "tipo_dato": tipo,
                "descripcion": desc,
                "datasets_presentes": "encuesta_comunidad_venezolana",
                "columnas_origen": f"encuesta_comunidad_venezolana:{c}",
            }
        )

    dicc_new = pd.DataFrame(dicc_rows)
    if DICC_PATH.exists():
        dicc_old = pd.read_csv(DICC_PATH)
        dicc_out = pd.concat([dicc_old, dicc_new], ignore_index=True)
        dicc_out = dicc_out.sort_values(["variable_estandar", "datasets_presentes"]).drop_duplicates(
            subset=["variable_estandar", "datasets_presentes"], keep="last"
        )
    else:
        dicc_out = dicc_new

    dicc_out.to_csv(DICC_PATH, index=False, encoding="utf-8")

    print(f"OK -> {OUT_DATASET}")
    print(f"Filas: {len(out):,} | Columnas: {len(out.columns)}")
    print(f"Diccionario actualizado -> {DICC_PATH}")


if __name__ == "__main__":
    main()
