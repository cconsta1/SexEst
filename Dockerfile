FROM python:3.7

EXPOSE 8501

WORKDIR /app

COPY . .
COPY models_goldman_missing_data /app/models_goldman_missing_data/
COPY models_howell_missing_data /app/models_howell_missing_data/
COPY .streamlit /app/.streamlit/

RUN pip3 install -r requirements.txt

CMD streamlit run streamlit_app.py