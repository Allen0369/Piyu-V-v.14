from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow import keras
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import uuid, os, json, torch, joblib, threading, time

app = Flask(__name__)
TEMP_RESULTS_DIR = "temp_results"
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

lstm_model = keras.models.load_model('models/model_lstm.keras')

def delete_file_later(path, delay=600):
    def _delete():
        time.sleep(delay)
        if os.path.exists(path):
            os.remove(path)
            print(f"[INFO] Deleted temp file: {path}")
    threading.Thread(target=_delete, daemon=True).start()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route('/results/<session_id>', methods=['GET'])
def get_results(session_id):
    file_path = os.path.join(TEMP_RESULTS_DIR, f"{session_id}.json")
    if not os.path.exists(file_path):
        return jsonify({"error": "Session ID not found"}), 404

    with open(file_path, "r") as f:
        result = json.load(f)

    return jsonify(result)
    
@app.route('/predict', methods=['POST'])
def predict():
    capacity = int(request.form['capacity'])
    units = int(request.form['units'])
    uploaded_files = request.files.getlist("files")
    
    if not uploaded_files:
        return jsonify({"error": "No files received"}), 400

    try:
        dfs = []

        for file in uploaded_files:
            filename = file.filename
            
            df = pd.read_csv(file)
            
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        
        # ----------------- Data Summary -----------------
        initial = combined_df.groupby('day_of_week').agg({
            'trip_distance': ['count', 'mean'],
            'passenger_count': 'sum',
            'trip_duration': 'mean'
        }).round(2)

        initial.columns = ['_'.join(col).strip() for col in initial.columns.values]
        initial = initial.reset_index()

        initial_data = initial.to_dict(orient='records')
        
        # ------------------ LSTM Model ------------------
        model_instance = ModelLSTM(combined_df, lstm_model)
        restructured_df = model_instance.restructureData(combined_df)
        X, y = model_instance.assignFeatureTarget(restructured_df)
        X_scaled, y_scaled, scaler_y = model_instance.scaleValues(X, y)
        y_pred_scaled = model_instance.predictLSTM(X_scaled, y_scaled)
        y_pred, _ = model_instance.scaleInverseTransform(scaler_y, y_pred_scaled, y_scaled)

        restructured_df['prediction'] = y_pred.flatten()
        
        # Janella's change
        valid_counts = restructured_df.groupby(['PULocationID', 'day_of_week']).size().reset_index(name='count')
        valid_pairs = valid_counts[valid_counts['count'] >= 2][['PULocationID', 'day_of_week']]
        filtered_for_viz = restructured_df.merge(valid_pairs, on=['PULocationID', 'day_of_week'])

        prediction_lstm = filtered_for_viz[['date', 'PULocationID', 'day_of_week', 'log_passenger_count_lag7', 'pct_change', 'prediction']].to_dict(orient='records')
        
        # ----------------- STGCN Model ------------------
        combined_with_preds = combined_df.merge(
            restructured_df[['date', 'PULocationID', 'day_of_week', 'prediction']],
            on=['date', 'PULocationID', 'day_of_week'],
            how='left'
        )
        
        stgc_data_generator = DataGeneratorSTGCN(combined_with_preds)
        num_zones, zone_ids, norm_adj, X_stgcn, y_stgcn, A_hat = stgc_data_generator.generateData(combined_with_preds)

        stgcn_model = STGCNRoute(num_nodes=num_zones)
        stgcn_model.eval()
        
        predicted_routes = stgcn_model(X_stgcn, A_hat).detach().numpy()[0]  # [N, N]
        recommended_routes = [{
            "origin": str(zone_ids[i]),
            "destination": [str(zone_ids[j]) for j in predicted_routes[i].argsort()[::-1][:5]]
            } for i in range(num_zones)]

        with torch.no_grad():
            stgcn_output = stgcn_model(X_stgcn, A_hat)
            
        # --------------- Allocation ------------------
        allocator = DynamicAllocator(alpha=0.8)
        allocator.set_user_inputs(capacity, units)
        allocations = allocator.allocate(combined_with_preds, stgcn_output, zone_ids)
        
        # ---------------- Results --------------------
        results = {
            "status": "success",
            "files_processed": [f.filename for f in uploaded_files],
            "lstm_results": prediction_lstm,
            "stgcn_results": recommended_routes,
            "initial": initial_data,
            "allocation_results": allocations.to_dict(orient='records')
        }
        
        session_id = str(uuid.uuid4())
        file_path = f"{TEMP_RESULTS_DIR}/{session_id}.json"
        with open(file_path, "w") as f:
            json.dump(results, f)

        delete_file_later(file_path, delay=900)

        return jsonify({"session_id": session_id})

    except Exception as e:
        return jsonify({'error': f'Error processing files: {str(e)}'}), 500

class ModelLSTM:
    def __init__(self, df, model_1):
        self.df = df
        self.model_1 = model_1

    def restructureData(self, df):
        res_df = df.groupby(['date', 'PULocationID', 'day_of_week'])['passenger_count'].sum().reset_index()
        res_df = res_df.sort_values(['PULocationID', 'day_of_week', 'date'])
        res_df['passenger_count_lag7'] = res_df.groupby(['PULocationID', 'day_of_week'])['passenger_count'].shift(1)
        res_df['pct_change'] = (res_df['passenger_count'] - res_df['passenger_count_lag7']) / (res_df['passenger_count_lag7'] + 1e-6)

        res_df = res_df.dropna(subset=['passenger_count', 'passenger_count_lag7'])
        res_df['passenger_count'] = res_df['passenger_count'].clip(lower=0)
        res_df['passenger_count_lag7'] = res_df['passenger_count_lag7'].clip(lower=0)

        res_df['log_passenger_count'] = np.log1p(res_df['passenger_count'])
        res_df['log_passenger_count_lag7'] = np.log1p(res_df['passenger_count_lag7'])

        return res_df

    def assignFeatureTarget(self, res_df):
        X = res_df[['log_passenger_count_lag7', 'pct_change']].values
        y = res_df['log_passenger_count'].values.reshape(-1, 1)

        return X, y

    def scaleValues(self, X, y):
        scaler_X = joblib.load('scaler_X.save')
        scaler_y = joblib.load('scaler_y.save')

        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y)

        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=1e6, neginf=-1e6)

        self.scaler_y = scaler_y

        return X_scaled, y_scaled, scaler_y

    def predictLSTM(self, X_scaled, y_scaled):
        y_pred_scaled = self.model_1.predict(X_scaled)

        return y_pred_scaled

    def scaleInverseTransform(self, scaler_y, y_pred_scaled, y_scaled):
        y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
        y_pred = np.expm1(y_pred_log)

        y_true = np.expm1(scaler_y.inverse_transform(y_scaled))

        return y_pred, y_true

class DataGeneratorSTGCN():
    def __init__(self, df_processed):
        self.df_processed = df_processed

    def zoneID(self, df_processed):
        zone_ids = sorted(df_processed['PULocationID'].unique())
        zone_to_idx = {zone: idx for idx, zone in enumerate(zone_ids)}
        num_zones = len(zone_ids)

        return num_zones, zone_ids, zone_to_idx

    def adjMatrix(self, df_processed, zone_to_idx, num_zones):
        adj_matrix = np.zeros((num_zones, num_zones))
        flows = pd.crosstab(df_processed['PULocationID'], df_processed['DOLocationID'])

        for pu in flows.index:
            for do in flows.columns:
                if pu in zone_to_idx and do in zone_to_idx:
                    adj_matrix[zone_to_idx[pu], zone_to_idx[do]] = flows.loc[pu, do]

        def normalize_adj(adj):
            A = adj + np.eye(adj.shape[0])
            D = np.diag(1 / np.sqrt(A.sum(1)))
            return D @ A @ D

        norm_adj = normalize_adj(adj_matrix)

        return norm_adj

    def tensorData(self, df_processed, zone_ids):
        features = df_processed.groupby(['PULocationID', 'day_of_week']).size().unstack(fill_value=0)
        features = features.reindex(zone_ids).fillna(0).to_numpy()
        features = (features - features.mean()) / features.std()

        X = torch.tensor(features.T[None, None, :, :], dtype=torch.float32)  # shape: (1, 1, 7, num_zones)

        flow_weighted_pred = df_processed.groupby(['PULocationID', 'DOLocationID'])['prediction'].sum().unstack(fill_value=0)
        flow_weighted_pred = flow_weighted_pred.reindex(index=zone_ids, columns=zone_ids, fill_value=0)
        flow_matrix_normalized = flow_weighted_pred.div(flow_weighted_pred.sum(axis=1).replace(0, 1), axis=0).to_numpy()
        y = torch.tensor(flow_matrix_normalized[None], dtype=torch.float32)

        return X, y

    def aHat(self, norm_adj):
        a_hat = torch.tensor(norm_adj, dtype=torch.float32)
        return a_hat

    def generateData(self, df):
        num_zones, zone_ids, zone_to_idx = self.zoneID(df)
        norm_adj = self.adjMatrix(df, zone_to_idx, num_zones)
        X, y = self.tensorData(df, zone_ids)
        A_hat = self.aHat(norm_adj)

        return num_zones, zone_ids, norm_adj, X, y, A_hat

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.temporal1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.graph_conv = nn.Linear(num_nodes, num_nodes)
        self.temporal2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.relu = nn.ReLU()

    def forward(self, x, A_hat):
        x = self.temporal1(x)
        x = torch.einsum("nctv,vw->nctw", x, A_hat)
        x = self.graph_conv(x)
        x = self.temporal2(x)
        return self.relu(x)

class STGCNRoute(nn.Module):
    def __init__(self, num_nodes, in_channels=1, hidden_channels=16):
        super().__init__()
        self.block = STGCNBlock(in_channels, hidden_channels, num_nodes)
        self.decoder = nn.Linear(hidden_channels, num_nodes)

    def forward(self, x, A_hat):
        x = self.block(x, A_hat)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        flow_pred = self.decoder(x)
        return torch.softmax(flow_pred, dim=-1)  # [B, N, N]

class DynamicAllocator:
    def __init__(self, alpha=0.8):
        self.alpha = alpha

    def set_user_inputs(self, capacity, units):
        self.vehicle_capacity = capacity
        default_units = units
        self.daily_units_dict = {day: default_units for day in range(7)}

    def allocate(self, df_with_preds, flow_pred, zone_ids):
        if not hasattr(self, 'vehicle_capacity') or not hasattr(self, 'daily_units_dict'):
            raise ValueError("You must call set_user_inputs() before allocate()")

        zone_ids = list(zone_ids)
        all_allocations = []

        for day in range(7):
            day_df = df_with_preds[df_with_preds['day_of_week'] == day]

            # Sum predicted demand per zone for that day
            predicted_demand = day_df.groupby('PULocationID')['prediction'].sum().reindex(zone_ids, fill_value=0)

            # Flow-based inflow
            inflow = pd.DataFrame(flow_pred[0].detach().numpy(), columns=zone_ids, index=zone_ids).sum(axis=0)

            # Combined score
            combined_score = self.alpha * predicted_demand + (1 - self.alpha) * inflow

            total_units = self.daily_units_dict.get(day, 0)
            total_capacity = total_units * self.vehicle_capacity

            zone_unit_allocation = np.ceil(predicted_demand / self.vehicle_capacity)

            allocation_day_df = pd.DataFrame({
                'zone': zone_ids,
                'predicted_demand': predicted_demand.values,
                'allocated_units': zone_unit_allocation.values.astype(int),
                'day_of_week': day
            })

            all_allocations.append(allocation_day_df)

        final_allocations = pd.concat(all_allocations, ignore_index=True)
        return final_allocations

if __name__ == "__main__":
    app.run(debug=True)