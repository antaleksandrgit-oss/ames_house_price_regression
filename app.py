from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ames_house_price_model.joblib"
SAMPLE_PATH = BASE_DIR / "sample_input.csv"


st.set_page_config(
    page_title="Ames House Price Prediction",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Ames House Price Prediction")
st.write(
    "Загрузите CSV-файл с характеристиками домов, "
    "чтобы получить прогноз стоимости."
)

try:
    model = load_model()
except Exception as error:
    st.error("Не удалось загрузить модель.")
    st.exception(error)
    st.stop()


st.download_button(
    label="Скачать пример CSV",
    data=SAMPLE_PATH.read_bytes(),
    file_name="sample_input.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader(
    "Загрузите CSV-файл",
    type="csv",
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error("Не удалось прочитать CSV-файл.")
        st.exception(error)
        st.stop()

    st.subheader("Загруженные данные")
    st.dataframe(data.head(), use_container_width=True)

    required_columns = list(model.feature_names_in_)
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        st.error(
            "В файле отсутствуют необходимые столбцы: "
            + ", ".join(missing_columns)
        )
        st.stop()

    model_input = data[required_columns].copy()

    if st.button("Рассчитать стоимость", type="primary"):
        try:
            predictions = model.predict(model_input)
        except Exception as error:
            st.error("Не удалось получить прогноз.")
            st.exception(error)
            st.stop()

        result = data.copy()
        result["PredictedSalePrice"] = predictions.round().astype(int)

        st.success(
            f"Прогноз рассчитан для {len(result)} домов."
        )

        st.subheader("Результаты")
        st.dataframe(
            result[["PredictedSalePrice"]],
            use_container_width=True,
        )

        st.download_button(
            label="Скачать результаты",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
