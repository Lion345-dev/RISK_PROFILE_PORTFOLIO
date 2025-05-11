# Perfil de Riesgo y Recomendación de Portafolios

Esta aplicación en Streamlit genera un perfil de riesgo personalizado para el usuario basado en un cuestionario y recomienda dos portafolios de inversión optimizados: uno con Alternativas Básicas y otro con Alternativas Premier. Utiliza datos financieros de APIs como Finviz, Yahoo Finance y Banxico, y sigue la metodología de optimización de portafolios de Markowitz y maximización del ratio de Sharpe.

## Características

* Cuestionario de perfilamiento para determinar el nivel de riesgo del usuario.
* Recomendación de dos portafolios personalizados: Básico y Premier.
* Visualizaciones interactivas de la asignación de activos y rendimiento histórico.
* Generación de un informe PDF con el perfil de riesgo y detalles de los portafolios.
* Soporte multilingüe (español e inglés) utilizando Grok.
* Advertencia sobre la diversificación temporal y análisis de escenarios.

## Requisitos

* Python 3.8+
* Dependencias listadas en `requirements.txt`
* Claves API para Finviz, Alpha Vantage, Grok y Banxico (configuradas en `.env`)

## Instalación

1. Clona el repositorio:

    ```bash
   git clone https://github.com/tu_usuario/risk_profile_portfolio.git
   cd risk_profile_portfolio
   ```

2. Crea un entorno virtual e instálalo:

   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:

   ```bas
   pip install -r requirements.txt
   ```

4. Configura las claves API en un archivo `.env`:

    ```TOKENS
   FINVIZ_API_KEY=tu_clave_finviz
   ALPHA_VANTAGE_API_KEY=tu_clave_alpha_vantage
   GROK_API_KEY=tu_clave_grok
   BANXICO_TOKEN=tu_token_banxico
   ```

## Uso

1. Ejecuta la aplicación:

   ```bash
   streamlit run app.py
   ```

2. Abre tu navegador en `http://localhost:8501` y completa el cuestionario de perfilamiento.
3. Selecciona las monedas deseadas y haz clic en "Calcular Portafolios".
4. Visualiza los portafolios recomendados y descarga el informe PDF.

## Estructura del Proyecto

* `app.py`: Lógica principal de la aplicación Streamlit.
* `data.py`: Funciones para extracción y preprocesamiento de datos.
* `portfolio.py`: Funciones para cálculo del perfil de riesgo y optimización de portafolios.
* `visuals.py`: Funciones para generar visualizaciones y el informe PDF.
* `.env`: Archivo para claves API (no versionado).
* `requirements.txt`: Lista de dependencias.

## Contribución

¡Las contribuciones son bienvenidas! Por favor, abre un issue o envía un pull request con tus mejoras.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](https://grok.com/chat/LICENSE) para más detalles.
