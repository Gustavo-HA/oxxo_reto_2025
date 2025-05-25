import pandas as pd
from config import (
    MES_VENTA_PATH,
    META_VENTA_PATH,
    RAW_TRAIN_DATA,
    RAW_TEST_DATA,
    INTERIM_DATA_DIR,
)


def create_labels():
    """
    Procesa los datos de ventas y tiendas para generar etiquetas de "EXITOSA"
    y guarda los datasets de entrenamiento y prueba con estas etiquetas.
    """
    # Cargar datos de ventas
    ventas = pd.read_csv(MES_VENTA_PATH)

    # Filtrar registros con valores de "Venta" útiles
    ventas = ventas[ventas["VENTA_TOTAL"] > ventas["VENTA_TOTAL"].quantile(0.005)]

    # Filtrar tiendas con al menos 12 registros utiles
    tiendas_a_retener = (ventas["TIENDA_ID"].value_counts() >= 12).to_dict()
    ventas["retener"] = ventas["TIENDA_ID"].map(tiendas_a_retener)
    ventas = ventas[ventas["retener"]]

    # Cargar diccionario de tipo: meta venta
    meta_venta = pd.read_csv(META_VENTA_PATH)
    meta_venta_dict = meta_venta.set_index('ENTORNO_DES')['Meta_venta'].to_dict()

    # Cargar diccionario de tienda_id : tipo para datos de entrenamiento
    tienda_tipo_train = pd.read_csv(RAW_TRAIN_DATA)
    tienda_tipo_train = tienda_tipo_train[["TIENDA_ID", "ENTORNO_DES"]]
    tienda_tipo_dict_train = tienda_tipo_train.set_index("TIENDA_ID")["ENTORNO_DES"].to_dict()

    # Cargar diccionario de tienda_id : tipo para datos de prueba
    tienda_tipo_test = pd.read_csv(RAW_TEST_DATA)
    tienda_tipo_test = tienda_tipo_test[["TIENDA_ID", "ENTORNO_DES"]]
    tienda_tipo_dict_test = tienda_tipo_test.set_index("TIENDA_ID")["ENTORNO_DES"].to_dict()

    # Combinar diccionarios de tipo de tienda
    tienda_tipo_dict_completo = tienda_tipo_dict_train.copy()
    tienda_tipo_dict_completo.update(tienda_tipo_dict_test)

    # Obtener el diccionario tienda_id : venta_meta
    tienda_meta_dict = dict()
    for tienda_id, tipo in tienda_tipo_dict_completo.items():
        if tipo in meta_venta_dict: # Asegurarse que el tipo existe en meta_venta_dict
            tienda_meta_dict[tienda_id] = meta_venta_dict[tipo]
        # else: # Opcional: manejar casos donde el tipo no tiene meta definida
            # print(f"Advertencia: Tipo '{tipo}' para tienda_id '{tienda_id}' no encontrado en meta_venta_dict.")

    ventas["meta"] = ventas["TIENDA_ID"].map(tienda_meta_dict)
    
    # Eliminar filas donde la meta no se pudo mapear (si alguna)
    ventas.dropna(subset=['meta'], inplace=True)


    ventas["cumple"] = (ventas["VENTA_TOTAL"] - ventas["meta"] >= 0).astype(int)

    # Calcular ratio de cumplimiento por tienda
    tienda_ratio = ventas.groupby("TIENDA_ID")["cumple"].mean().reset_index()
    tienda_ratio["EXITOSA"] = (tienda_ratio["cumple"] >= 0.6).astype(int)

    tienda_exitosa_dict = tienda_ratio.set_index("TIENDA_ID")["EXITOSA"].to_dict()

    # Cargar datasets de tiendas para añadir la etiqueta "EXITOSA"
    dataset_train = pd.read_csv(RAW_TRAIN_DATA)
    dataset_test = pd.read_csv(RAW_TEST_DATA)

    dataset_train["EXITOSA"] = dataset_train["TIENDA_ID"].map(tienda_exitosa_dict)
    dataset_test["EXITOSA"] = dataset_test["TIENDA_ID"].map(tienda_exitosa_dict)

    # Filtrar registros donde la etiqueta "EXITOSA" no pudo ser asignada (si aplica)
    dataset_train = dataset_train[dataset_train["EXITOSA"].notna()]
    dataset_test = dataset_test[dataset_test["EXITOSA"].notna()]
    
    # Convertir la columna EXITOSA a entero por si acaso tiene NaNs que la convirtieron a float
    if "EXITOSA" in dataset_train.columns:
        dataset_train["EXITOSA"] = dataset_train["EXITOSA"].astype(int)
    if "EXITOSA" in dataset_test.columns:
        dataset_test["EXITOSA"] = dataset_test["EXITOSA"].astype(int)


    # Guardar los datasets con la nueva etiqueta
    output_train_path = INTERIM_DATA_DIR / "train_w_label.csv"
    output_test_path = INTERIM_DATA_DIR / "test_w_label.csv"

    # Asegurarse de que el directorio de salida exista (opcional, INTERIM_DATA_DIR ya es un Path)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_train.to_csv(output_train_path, index=False)
    dataset_test.to_csv(output_test_path, index=False)

    print(f"Archivo de entrenamiento con etiquetas guardado en: {output_train_path}")
    print(f"Archivo de prueba con etiquetas guardado en: {output_test_path}")
    print("\nConteos de la etiqueta 'EXITOSA' en el dataset de entrenamiento:")
    print(dataset_train["EXITOSA"].value_counts())
    print("\nConteos de la etiqueta 'EXITOSA' en el dataset de prueba:")
    print(dataset_test["EXITOSA"].value_counts())


if __name__ == "__main__":
    create_labels()

