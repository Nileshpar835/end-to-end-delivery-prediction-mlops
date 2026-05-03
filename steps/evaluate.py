from zenml import step
import torch

@step
def evaluate_model(model, X_test, y_test):

    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values.reshape(-1,1), dtype=torch.float32)

    with torch.no_grad():
        pred = model(X_test)
        loss = torch.nn.MSELoss()(pred, y_test)

    print("Test Loss:", loss.item())