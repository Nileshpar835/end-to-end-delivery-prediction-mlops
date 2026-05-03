from zenml import step
import torch
import mlflow
import os
import joblib

from model.model import get_nn_model, get_linear_model


@step
def train_model(X_train, y_train):

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

    input_dim = X_train_tensor.shape[1]
    print(f"Input Dimension: {input_dim}")

    linear_model = get_linear_model(input_dim)
    nn_model = get_nn_model(input_dim)

    def train(model):
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(200):
            optimizer.zero_grad()
            preds = model(X_train_tensor)
            loss = loss_fn(preds, y_train_tensor)
            loss.backward()
            optimizer.step()

        return loss.item()

    with mlflow.start_run():

        linear_loss = train(linear_model)
        nn_loss = train(nn_model)

        print(f"Linear Loss: {linear_loss}")
        print(f"NN Loss: {nn_loss}")

        mlflow.log_metric("linear_loss", linear_loss)
        mlflow.log_metric("nn_loss", nn_loss)

        # Select best model
        best_model = nn_model if nn_loss < linear_loss else linear_model

        # Save model (state_dict)
        model_path = os.path.join(os.getcwd(), "model.pth")
        torch.save(best_model.state_dict(), model_path)

        # 🔥 Save feature columns (VERY IMPORTANT)
        columns_path = os.path.join(os.getcwd(), "columns.pkl")
        joblib.dump(X_train.columns.tolist(), columns_path)

        print(f"Model saved at: {model_path}")
        print(f"Columns saved at: {columns_path}")

        # Log model in MLflow
        mlflow.pytorch.log_model(best_model, "model")

    return best_model