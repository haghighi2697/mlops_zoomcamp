import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Define the experiment name where the results are saved
experiment_name = "random-forest-best-models"

# Create an MlflowClient instance
client = MlflowClient()

# Get the experiment ID
experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

# Search for the top 5 runs in the experiment
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.rmse ASC"]
)

# Check if there are any runs
if not runs:
    print("No runs found in the experiment.")
    exit()

# Initialize variables to track the best run
best_run = None
lowest_rmse = float('inf')

# Calculate RMSE for each run and find the best one
for run in runs:
    rmse = run.data.metrics['rmse']
    if rmse < lowest_rmse:
        lowest_rmse = rmse
        best_run = run

# Ensure we have a best run
if not best_run:
    print("No suitable run found.")
    exit()

# Get the run ID and model URI of the best run
best_run_id = best_run.info.run_id
model_uri = f"runs:/{best_run_id}/model"

# Define the model name for registration
model_name = "best_random_forest_model"

# Register the best model
mlflow.register_model(model_uri=model_uri, name=model_name)

print(f"Model registered successfully. Model name: {model_name}, Run ID: {best_run_id}")
