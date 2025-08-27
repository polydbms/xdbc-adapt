import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor, DMatrix
import Transfer_Data_Processor
from Configs import *

class Per_Environment_RF_Cost_Model:

    def __init__(self,
                 input_fields,
                 metric='time',
                 data_per_env=100,
                 underlying="cost_model",
                 cluster=True,
                 regression_model='xgb',
                 network_transformation=True,
                 history_ratio=0.5,
                 history_weight_decay_factor=0.9):
        """
        Initializes the Per_Environment_RF_Cost_Model, which trains and predicts using
        Regression Models for multiple environments or clusters.

        Parameters:
            input_fields (list): List of feature names used for training and prediction.
            metric (str): Metric to optimize.
            underlying (str): Identifier-String for the underlying model.
            cluster (bool): True if data should be clustered, False if one model per environment
            regression_model (str): The underlying regression model to use [rfs, gdb, xgb]
            network_transformation (bool): True if network values should be transformed with sigmoid like function for distance calculation. Has no effect if history_ratio is 1.
            history_ratio (float): The ratio between the weights calculated from the environment signatures, and from the weight history. Should be between [0,1]. 0 meaning only weights from environments signatures, and 1 meaning only weights from history.
            history_weight_decay_factor (float): The decay factor with which to calculate the history weights.
         """
        self.input_fields = input_fields
        self.metric = metric
        self.underlying = underlying
        self.data_per_env = data_per_env
        self.cluster = cluster
        self.regression_model = regression_model
        self.network_transformation = network_transformation
        self.history_ratio = history_ratio
        self.history_weight_decay_factor = history_weight_decay_factor

        self.continous_maintained_history_vector = None
        self.norm_total = 0
        self.total_history_updates = 0
        self.models = {}  # dict to store models's per environment
        self.environments = []  # list of known environments
        self.datasets = {}
        self.env_features = {}  # each environment’s [server_cpu, client_cpu, network]

        self.compression_mapping = {"nocomp": 0, "zstd": 1, "lz4": 2, "lzo": 3, "snappy": 4, "no_comp":0}
        self.reverse_compression_mapping = {v: k for k, v in self.compression_mapping.items()}

    def train(self, x_train, y_train):
        """
        Train the regression models. Splits data per Environment, and trains one model for each environment.

        Parameters:
           x_train (dataframe): Input data containing all fields from input fields.
           y_train (dataframe): Target data containing the values of the specified metric.
        """

        combine_data = x_train.copy()
        combine_data[self.metric] = y_train

        combine_data = self.convert_dataframe(combine_data)

        if self.cluster:
            # cluster data automatically
            grouped = Transfer_Data_Processor.process_data(data=combine_data, training_data_per_env=self.data_per_env, cluster_labes_avg=True, n_clusters=0)
        else:
            # one cluster per environment
            grouped = Transfer_Data_Processor.process_data(data=combine_data, training_data_per_env=self.data_per_env, cluster_labes_avg=True, n_clusters=-1)


        # then train a model for each environment-group
        for env, group in grouped.items():

            env = env.replace("cluster-avg_","")

            X = group[self.input_fields].values
            y = group[self.metric].values

            if self.regression_model == 'rfs':
                model_name = "RandomForestRegressor"
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=5,#2,
                    min_samples_leaf=4,#1,
                    #bootstrap=True,
                    random_state=123)
            elif self.regression_model == 'xgb':
                model_name = "XGBRegressor"
                model = XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=100,#300
                    max_depth=None,
                    learning_rate=0.05, #0.1
                    min_child_weight=1,
                    reg_lambda=1,
                    reg_alpha=0,
                    gamma=0.5,#0.2
                    #colsample_bytree=0.8,
                    subsample=0.8,
                    random_state=123)
            else:
                raise ValueError(f"Unkown underlying regression model: {self.regression_model}")


            model.fit(X, y)

            self.models[env] = model
            self.datasets[env] = group
            self.environments.append(env)

            env_dict = unparse_environment_float(env)
            self.env_features[env] = [
                round(env_dict['server_cpu'], 2),
                round(env_dict['client_cpu'], 2),
                round(env_dict['network'], 2)
            ]

            print(f"Trained {model_name} for cluster {env} with length {len(group)}")

        self.continous_maintained_history_vector = np.zeros(len(self.models))
        self.continous_maintained_constant_vector = np.zeros(len(self.models))
        self.len_history = 0



    def update(self, config, result):
        """
        Updates the weight history by calculating a weight vector for the  configuration
        and result,depending on how close they are to the individual models' predictions.

        Parameters:
            config (dict): Configuration which was executed.
            result (dict): Result of that execution contianing `self.metric`.
        """

        data = self.convert_dict(dict(config))

        X = np.array([[data[field] for field in self.input_fields]], dtype=float)

        # Collect all prediction of the known models
        known_predictions = []
        for env, model in self.models.items():
            pred = model.predict(X)[0]
            known_predictions.append(pred)
        known_predictions = np.array(known_predictions)

        # Calculate the difference between the true result and the predictions
        errors = (known_predictions - result[self.metric]) ** 2
        weights = 1 / (errors + 1e-8)
        weights_normalized = weights / np.sum(weights)

        # Calculate constant error term
        const = (known_predictions - result[self.metric])
        self.len_history = self.len_history + 1
        #self.continous_maintained_constant_vector = (self.continous_maintained_constant_vector * ( (self.len_history - 1) / self.len_history )) + (const * (1 / self.len_history))


        #ratio = 0.8
        #if self.len_history == 0:
        #    ratio = 0
        #self.continous_maintained_constant_vector = (self.continous_maintained_constant_vector * (ratio)) + (const * (1 - ratio))



        self.norm_total = self.history_weight_decay_factor * (self.norm_total + 1)
        self.total_history_updates = self.total_history_updates + 1
        self.continous_maintained_history_vector = (self.history_weight_decay_factor * self.continous_maintained_history_vector) + weights_normalized
        self.continous_maintained_history_vector = self.continous_maintained_history_vector / self.norm_total
        self.continous_maintained_history_vector = self.continous_maintained_history_vector / sum(self.continous_maintained_history_vector)

    def convert_dataframe(self, df, reverse=False):
        field_name = "compression" if "compression" in df.columns else "compression_lib"
        mapping = self.reverse_compression_mapping if reverse else self.compression_mapping
        df[field_name] = df[field_name].map(mapping).fillna(df[field_name])
        return df

    def convert_dict(self, data, reverse=False):
        field_name = "compression" if "compression" in data else "compression_lib"
        mapping = self.reverse_compression_mapping if reverse else self.compression_mapping
        data[field_name] = mapping.get(data.get(field_name, None), data.get(field_name, None))
        return data

    def predict(self, data, target_environment, print_wieghts=False):
        """
        Predict for a target environment using the cost model for that environment,
        or if the environment is not known, a linear combination of known models.

        Parameters:
            data (dict): Data to be predicted on.
            target_environment (dict): Dictionary containing the values for the target environment (server_cpu,client_cpu,network).
        Returns:
            dict: Dictionary containing the result of the specified metric.
        """

        data = self.convert_dict(dict(data))
        #if isinstance(data, dict):
        X = np.array([[data[field] for field in self.input_fields]], dtype=float)
        #elif isinstance(data, pd.DataFrame):
        #    X = data[self.input_fields].values
        #else:
        #    X = np.asarray(data)

        target_server = target_environment['server_cpu']
        target_client = target_environment['client_cpu']
        target_network = target_environment['network']
        target_features = np.array([[target_server, target_client, target_network]], dtype=float)



        # if the target environment is known, use its model directly #todo temporary disable, we dont test this anyway
        #if environment_to_string(target_environment) in self.environments:
        #    y = self.models[environment_to_string(target_environment)].predict(X)
        #    return {self.metric: y[0]}

        env_keys = list(self.env_features.keys())
        known_features = np.array([self.env_features[env] for env in env_keys], dtype=float)

        # Collect all known environments and their predictions
        known_predictions = []
        for env in env_keys:
            model = self.models[env]
            pred = model.predict(X)[0]
            known_predictions.append(pred)
        known_predictions = np.array(known_predictions)


        # Transform network using a sigmoid like function
        if self.network_transformation:
            def transform_network(x):
                return np.round(20 / (1 + 31 * np.exp(-0.02 * x)), 2)
            known_features[:, 2] = transform_network(known_features[:, 2])
            target_features[0, 2] = transform_network(target_features[0, 2])
        else:
            known_features[:, 2] /= 100
            target_features[0, 2] /= 100


        # Calculate the distances between environment signatures
        distances = np.linalg.norm(known_features - target_features[0], axis=1) ** 1.5
        distances = np.where(distances == 0, 1e-8, distances) # to not divide by 0
        environment_weights = 1 / distances
        environment_weights_normalized = environment_weights / np.sum(environment_weights)


        # Use a combination of both weights or only environment weights
        if self.history_ratio == 0:
            combined_weights = environment_weights_normalized

        elif self.history_ratio < 1:

            # Dynamic history-ratio 
            if self.total_history_updates == 0:
                hist_weight_ratio = 0
            elif self.history_ratio == 0.5:
                hist_weight_ratio = 0.5
            else:
                #hist_weight_ratio = self.total_history_updates / ((self.total_history_updates * 1.0) + 1)
                hist_weight_ratio = min(1, np.log(self.total_history_updates + 1) / np.log(21))


            combined_weights = (self.continous_maintained_history_vector * hist_weight_ratio) + (environment_weights_normalized * (1 - hist_weight_ratio))

            # Combine the environment & history weight vectors based on history_ratio
            #combined_weights = (self.continous_maintained_history_vector * self.history_ratio) + (environment_weights_normalized * (1 - self.history_ratio))

        # Use only history weights
        elif self.history_ratio == 1:
            # Edge case if history is empty
            if np.all(self.continous_maintained_history_vector == 0):

                #if target_environment is not None:
                #    combined_weights = environment_weights_normalized
                #else:
                combined_weights = np.full(self.continous_maintained_history_vector.shape, 1/self.continous_maintained_history_vector.size)
            else:
                combined_weights = self.continous_maintained_history_vector / sum(self.continous_maintained_history_vector)

        # Normalize weight vector to 1
        combined_weights_normalized = combined_weights / np.sum(combined_weights)

        # Multiply predictions with weights to get final prediction
        prediction_w_history = np.dot(combined_weights_normalized, known_predictions.flatten())


        predictions_with_constant = known_predictions.flatten()# permant disable - self.continous_maintained_constant_vector
        prediction_w_constant = np.dot(combined_weights_normalized, predictions_with_constant)
        '''
        if print_wieghts:
            #use normalized weights for combining predictions, non normalized for environments printing

            prediction = np.dot(environment_weights_normalized, known_predictions.flatten())
            predicted_environment = np.dot(environment_weights, known_features)

            predicted_environment_w_history = np.dot(combined_weights, known_features)

            data_df = {'Env/Cluster': [" ".join(map(str, row)) for row in known_features],
                       'Predictions': known_predictions[0],
                       'Weights': environment_weights_normalized,
                       'Weights w/ hist.': combined_weights_normalized,
                       'Hist. Weights': self.continous_maintained_history_vector}

            df_to_print = pd.DataFrame(data_df)
            df_to_print = df_to_print.round(2)

            df_to_print = df_to_print.sort_values(by='Weights w/ hist.', ascending=True, inplace=False)

            table = PrettyTable()
            table.field_names = df_to_print.columns.tolist()
            for row in df_to_print.itertuples(index=False):
                table.add_row(row)

            print(f"\nTarget Environment: S{target_environment['server_cpu']}_C{target_environment['client_cpu']}_N{target_environment['network']}")
            print(f"Feature vector: {target_features}")
            print(table)

            print(f"Prediction :         { round(prediction,2)}")
            print(f"Prediction w/ hist.: { round(prediction_w_history,2)}")
            print(f"Estimated Environment :         { np.round(predicted_environment,2)}")
            print(f"Estimated Environment w/ hist.: { np.round(predicted_environment_w_history,2)}")
        '''
        #print_wieghts = False
        if print_wieghts:
            table = PrettyTable()
            table.field_names = ['Env/Cluster', 'Prediction', 'Env Weight', 'Combined Weight', 'History Weight', 'Constant Vector', 'Prediction + Constant']
            for i, env in enumerate(env_keys):
                table.add_row([
                    env,
                    round(known_predictions[i], 2),
                    round(environment_weights_normalized[i], 2),
                    round(combined_weights_normalized[i], 2),
                    round(self.continous_maintained_history_vector[i], 2),
                    round(self.continous_maintained_constant_vector[i],2),
                    round(predictions_with_constant[i],2)
                ])
            print(f"\nTarget Environment: S{target_server}_C{target_client}_N{target_network}")
            print("Feature vector:", target_features)
            print(table)
            weighted_pred = np.dot(environment_weights_normalized, known_predictions.flatten())
            print(f"Prediction (env weights): {round(weighted_pred, 2)}")
            print(f"Prediction (combined weights): {round(prediction_w_history, 2)}")
            print(f"Prediction (with constant ): {round(prediction_w_constant, 2)}")

        return {self.metric: prediction_w_constant}



    def get_predictions_vector(self, data):
        data = self.convert_dict(dict(data))
        if isinstance(data, dict):
            X = np.array([[data[field] for field in self.input_fields]], dtype=float)
        elif isinstance(data, pd.DataFrame):
            X = data[self.input_fields].values
        else:
            X = np.asarray(data)

        env_keys = list(self.env_features.keys())

        # Collect all known environments and their predictions
        known_predictions = []
        for env in env_keys:
            model = self.models[env]
            pred = model.predict(X)[0]
            known_predictions.append(pred)
        known_predictions = np.array(known_predictions)

        return known_predictions.flatten()




    def predict_batch_01(self, data_list, target_environment):
        X_list = []
        for data in data_list:
            converted_data = self.convert_dict(dict(data))
            try:
                features = [converted_data[field] for field in self.input_fields]
                X_list.append(features)
            except KeyError as e:
                raise ValueError(f"Input data missing required field: {e}. Data: {data}") from e

        X = np.array(X_list, dtype=float)
        #num_predictions = X.shape[0] # Number of configurations in the batch

        target_server = target_environment['server_cpu']
        target_client = target_environment['client_cpu']
        target_network = target_environment['network']
        target_features = np.array([[target_server, target_client, target_network]], dtype=float)


        env_keys = list(self.env_features.keys())
        known_features = np.array([self.env_features[env] for env in env_keys], dtype=float)

        known_predictions_batch = []
        for env in env_keys:
            model = self.models[env]
            pred_batch = model.predict(X)
            known_predictions_batch.append(pred_batch)
        known_predictions_batch = np.stack(known_predictions_batch, axis=0)

        if self.network_transformation:
            def transform_network(x):
                return np.round(20 / (1 + 31 * np.exp(-0.02 * x)), 2)
            known_features[:, 2] = transform_network(known_features[:, 2])
            target_features[0, 2] = transform_network(target_features[0, 2])
        else:
            known_features[:, 2] = known_features[:, 2] / 100.0
            target_features[0, 2] = target_features[0, 2] / 100.0


        distances = np.linalg.norm(known_features - target_features[0], axis=1) ** 1.5
        distances = np.where(distances == 0, 1e-8, distances) # Avoid division by zero
        environment_weights = 1 / distances
        sum_env_weights = np.sum(environment_weights)
        if sum_env_weights == 0:
            environment_weights_normalized = np.full(len(env_keys), 1.0 / len(env_keys))
        else:
            environment_weights_normalized = environment_weights / sum_env_weights

        if self.history_ratio == 0:
            combined_weights = environment_weights_normalized
        elif self.history_ratio < 1:
            if self.total_history_updates == 0:
                hist_weight_ratio = 0
            elif self.history_ratio == 0.5:
                hist_weight_ratio = 0.5
            else:
                hist_weight_ratio = min(1, np.log(self.total_history_updates + 1) / np.log(21))

            history_vector_normalized = self.continous_maintained_history_vector / np.sum(self.continous_maintained_history_vector) if np.sum(self.continous_maintained_history_vector) > 0 else np.full_like(self.continous_maintained_history_vector, 1.0/len(self.continous_maintained_history_vector))

            combined_weights = (history_vector_normalized * hist_weight_ratio) + \
                               (environment_weights_normalized * (1 - hist_weight_ratio))
        elif self.history_ratio == 1:
            history_sum = np.sum(self.continous_maintained_history_vector)
            if history_sum == 0:
                combined_weights = np.full(len(env_keys), 1.0 / len(env_keys))
            else:
                combined_weights = self.continous_maintained_history_vector / history_sum

        sum_combined_weights = np.sum(combined_weights)
        if sum_combined_weights == 0:
            combined_weights_normalized = np.full(len(env_keys), 1.0 / len(env_keys))
        else:
            combined_weights_normalized = combined_weights / sum_combined_weights


        final_predictions = np.dot(combined_weights_normalized, known_predictions_batch)

        results_list = []
        for original_config, prediction_value in zip(data_list, final_predictions):
            results_list.append((original_config, float(prediction_value)))

        return results_list


