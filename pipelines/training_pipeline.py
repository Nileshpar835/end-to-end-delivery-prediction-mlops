from zenml import pipeline
from steps.ingest import ingest_data
from steps.preprocess import preprocess_data
from steps.train import train_model
from steps.evaluate import evaluate_model


@pipeline
def training_pipeline():
    df = ingest_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


    